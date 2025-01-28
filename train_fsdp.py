import os
import fire
import json
import time
import numpy as np
from dataclasses import asdict
from contextlib import nullcontext
from functools import partial
from llama import LLaMAConfig, LLaMA, LLaMABlock, Fp8LLaMA, Fp8LLaMABlock
from mistral import MistralConfig, Mistral, MistralBlock, Fp8Mistral, Fp8MistralBlock

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)

from torch.utils.data import DataLoader,IterableDataset
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp

class RandData(IterableDataset):
    def __init__(self, vocab_size, max_seq_len, total_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.total_size = total_size
    
    def __len__(self):
        return self.total_size
    
    def __iter__(self):
        datas = []
        for i in range(self.total_size):
            input = torch.randint(self.vocab_size, [self.max_seq_len], dtype=torch.int64)
            label = torch.cat([input[:-1], torch.randint(self.vocab_size, [1])])
            datas.append((input,label))
        return iter(datas)

def create_dummy_data_loader(world_size, batch_size, num_iteration, model_config):
    dataset = RandData(model_config.vocab_size, model_config.max_seq_len, batch_size*num_iteration*world_size)
    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=world_size, pin_memory=True, shuffle=False
    )
    return data_loader

def train(
    config_file: str,
    model_name: str,
    batch_size: int = 8,
    num_iteration: int = 128,
    grad_accumlate_pre_steps: int = 8, # steps to accumlate gradient
    reduce_pre_steps: int = 32, # steps to do all reduce
    enable_fp8: bool = False,
    enable_compile: bool = True,
    seed: int = 1024 # to ensure reproducible
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    assert rank == local_rank , "This script is intended to run on single node for testing"
    world_size = int(os.environ["WORLD_SIZE"])
    if enable_fp8:
        enable_compile = False
        if rank == 0:
            print('PyTorch compile currently doesn\'t work with Transformer Engine.')
    # Construct process group
    if local_rank == 0:
        print("Initing communication")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    assert dist.get_rank() == local_rank
    # Configure training setup
    if local_rank == 0:
        print("Using config",config_file)
    with open(config_file) as f:
        config = json.load(f)
    
    if model_name == "llama":
        model_config = LLaMAConfig(**config)
    elif model_name == "mistral":
        model_config = MistralConfig(**config)
    else:
        print("model not supported. please pass either llama or mistral as param")
    if local_rank == 0:
        print("Creating model with config: ",model_config)

    if enable_fp8: # add more model
        if model_name == "llama":
            layer_class = Fp8LLaMABlock
            model = Fp8LLaMA(**asdict(model_config))
        elif model_name == "mistral":
            layer_class = Fp8MistralBlock
            model = Fp8Mistral(**asdict(model_config))  
    else:
        if model_name == "llama":
            layer_class = LLaMABlock
            model = LLaMA(**asdict(model_config))
        elif model_name == "mistral":
            layer_class = MistralBlock
            model = Mistral(**asdict(model_config))

    model_config.estimate_flops_per_token(model,batch_size)  # Need to calculate before wrapping in FSDP

    if local_rank == 0:
        print(f"Loaded model on CPU with number of parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B") 
        print(f"Original Model:\n{model}")

    # FSDP
    model = FSDP(
        model,
        device_id=local_rank,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_class}),
        use_orig_params=True
    )
    if enable_fp8:
        prepare_te_modules_for_fsdp(model)
        fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
        all_worker = dist.new_group(backend='nccl')

    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    # Print out allocated device memory 
    pre_mem_use = torch.cuda.memory_allocated(device=f"cuda:{local_rank}") * 1e-6
    flops_per_iter = model_config.flops_per_token * (batch_size * model_config.max_seq_len)
    if local_rank == 0:
        print(f"GPU memory use = {pre_mem_use}MB")
        print("TFLOP per iteration:",flops_per_iter/1e12)
    # PyTorch compile
    if enable_compile:
        if rank == 0:
            print(f'Compiling model....')
        model = torch.compile(model)

    ddp_loss = torch.zeros(2, device=local_rank)
    model.train()
    iter_times = []
    warm_up = num_iteration/2

    data_loader = create_dummy_data_loader(world_size, batch_size, num_iteration, model_config)
    last_time = time.time()
    
    for step_idx, data_batch in enumerate(data_loader):
        input, labels = data_batch
        input = input.to(local_rank)
        labels = labels.to(local_rank)
        fp8_context =  nullcontext() if not enable_fp8 else te.fp8_autocast(enabled=enable_fp8, fp8_recipe=fp8_recipe, fp8_group=all_worker)
        with torch.amp.autocast('cuda', torch.bfloat16), fp8_context:
            weight_cache = enable_fp8 and (step_idx % grad_accumlate_pre_steps == 0)
            logits = model(input, is_first_microbatch=weight_cache)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            loss /= grad_accumlate_pre_steps

        loss.backward()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += input.size(0)

        if (step_idx + 1) % grad_accumlate_pre_steps == 0:
            # https://github.com/foundation-model-stack/fms-fsdp/blob/0fdb43dcfd31ab093f8d873b58b0b531dd0818b1/fms_fsdp/utils/train_utils.py#L94
            # https://github.com/foundation-model-stack/foundation-model-stack/blob/d55a9f2ade65ef4157cdfd928300874e2348e5d0/fms/training/trainer.py#L36
            model.clip_grad_norm_(1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (step_idx + 1) % reduce_pre_steps == 0:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        if rank == 0:
            current_time = time.time()
            iter_time = current_time-last_time
            token_per_sec = (batch_size * model_config.max_seq_len)/iter_time
            if step_idx > warm_up:
                print(f"Step: {step_idx}; TFLOP/s: {flops_per_iter/iter_time/1e12}; iteration time: {iter_time}; token per second: {token_per_sec}")
                iter_times.append(iter_time)
            else:
                print(f"warming up iter: {step_idx}/{warm_up}; TFLOP/s: {flops_per_iter/iter_time/1e12}; iteration time: {iter_time}; token per second: {token_per_sec}")
            last_time = current_time
        
        if (step_idx+1) == num_iteration:
            break

    if rank == 0:
        iter_times = np.array(iter_times)
        avg_iter_time = np.mean(iter_times)
        print("Avg token per second:",(batch_size * model_config.max_seq_len)/avg_iter_time)
        print("Avg iter time:",avg_iter_time)
        print("TFLOP per iteration:",flops_per_iter/1e12)
        print("Avg TFLOP/s,",flops_per_iter/avg_iter_time/1e12)
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}") * 1e-6
        print(f"Peak memory use = {peak_memory}MB")

    torch.cuda.empty_cache()
    dist.barrier()
    destroy_process_group()

if __name__ == '__main__':
    fire.Fire(train)
