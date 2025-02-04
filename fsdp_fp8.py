# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import json
import argparse
import subprocess
import pandas as pd

from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import always_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

# RNG state tracker for checkpointing
rng_seed = 1234
torch.manual_seed(rng_seed)
torch.cuda.manual_seed(rng_seed)
CUDA_RNG_STATES_TRACKER = te.distributed.CudaRNGStatesTracker()
CUDA_RNG_STATES_TRACKER.add("model-parallel-rng", rng_seed)
import time

class CustomDataset(Dataset):
    def __init__(self, datapaths='../data/tiny_text/tiny-textbooks/train-00000-of-00001.parquet', tokenizer=None, max_length=512):
        # Load the dataset from Hugging Face
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = pd.read_parquet(datapaths)['textbook']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the input and label for the specified index
        text = self.dataset[idx]*1000
        # print(text)
        # text = item['text']  # Change 'text' to your input column name
        label = text[1:]  # Change 'label' to your label column name

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = self.tokenizer.encode_plus(
            label,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Return the input IDs and attention mask, along with the label
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove the batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label['input_ids'].squeeze(),  # Remove the batch dimension
        }


class CustomDataLoader:
    def __init__(self, batch_size=32, tokenizer=None, max_length=512, shuffle=True):
        self.dataset = CustomDataset(tokenizer=tokenizer, max_length=max_length)
        num_tasks = WORLD_SIZE
        global_rank = LOCAL_RANK
        sampler_train = torch.utils.data.DistributedSampler(
            self.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        self.dataloader = DataLoader(self.dataset, sampler=sampler_train, batch_size=batch_size)

    def get_dataloader(self):
        return self.dataloader


def get_cuda_rng_tracker():
    return CUDA_RNG_STATES_TRACKER


def apply_fsdp_checkpointing(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    wrapper = lambda m: checkpoint_wrapper(
        m,
        checkpoint_fn=te.distributed.checkpoint,
        use_reentrant=False,
        get_rng_state_tracker=get_cuda_rng_tracker,
    )
    check_fn = lambda submodule: isinstance(submodule, blocks)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)


def lowercase(s):
    return str(s).lower()


def all_reduce_mean(x):
    world_size = WORLD_SIZE
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
    

def torch_dtype(d):
    typemap = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if lowercase(d) not in typemap.keys():
        raise TypeError
    return typemap[lowercase(d)]


te_layer_map = {
    "linear": te.Linear,
    "layernorm": te.LayerNorm,
    "rmsnorm": te.RMSNorm,
    "layernormlinear": te.LayerNormLinear,
    "layernormmlp": te.LayerNormMLP,
    "multiheadattention": te.MultiheadAttention,
    "transformerlayer": te.TransformerLayer,
}


def estimate_flops_per_token(args):
    embedding_dim = args.head_dim * args.num_heads
    head_dim = embedding_dim // args.num_heads

    """Calculate training TFLOP"""
    ffn1_flops = (
        2
        * args.batch_size
        * args.seq_length
        * args.ffn_hidden_size
        * embedding_dim
        * 2 # len(config.mlp_activations)
    )
    ffn2_flops = 2 * args.batch_size * args.seq_length * args.ffn_hidden_size * embedding_dim
    total_ffn_flops = ffn1_flops + ffn2_flops

    qkv_flops = (
        2
        * args.batch_size
        * args.seq_length
        * embedding_dim
        * (args.num_heads + 2 * args.num_kv)
        * head_dim
    )
    attention_flops = 4 * args.batch_size * args.seq_length**2 * args.num_heads * head_dim
    projection_flops = (
        2 * args.batch_size * args.seq_length * embedding_dim * args.num_heads * head_dim
    )
    embedding_flops = 2 * args.batch_size * args.seq_length * embedding_dim * args.vocab_size

    # multiply by 3 for both feed forward and back propagation flops
    learnable_weight_tflops = (
        ((total_ffn_flops + qkv_flops + projection_flops) * args.num_layers + embedding_flops) * 3
    )

    # megatron tflops calculation does not account for causality in attention
    attention_tflops = attention_flops * args.num_layers * 3

    total_tflops = learnable_weight_tflops + attention_tflops
    flops_per_token =  total_tflops# / (args.batch_size * args.seq_length)
    return flops_per_token

def te_layer(l):
    if l is not None:
        if lowercase(l) not in te_layer_map.keys():
            raise TypeError
        return te_layer_map[lowercase(l)]
    return None


def get_layer_args(args, layer_type=None):
    hidden_size = args.num_heads * args.head_dim
    layer_args = (hidden_size,)
    layer_kwargs = {
        "params_dtype": args.dtype,
        "device": "cuda" if args.no_defer_init else "meta",
        "get_rng_state_tracker": get_cuda_rng_tracker,
    }
    if layer_type is None:
        layer_type = args.layer_type

    if layer_type in [te.Linear, te.LayerNormLinear, te.LayerNormMLP]:
        ffn_hidden_size = int(3.5 * hidden_size) if args.num_layers == 1 else hidden_size
        layer_args += (ffn_hidden_size,)
        layer_kwargs["bias"] = True
        if layer_type == te.LayerNormMLP:
            layer_kwargs["seq_length"] = args.seq_length
    elif layer_type == te.MultiheadAttention:
        layer_args += (args.num_heads,)
        layer_kwargs["fuse_qkv_params"] = True
        layer_kwargs["input_layernorm"] = True
    elif layer_type == te.TransformerLayer:
        layer_args += (int(3.5 * hidden_size), args.num_heads)
        layer_kwargs["kv_channels"] = args.num_kv
        layer_kwargs["fuse_qkv_params"] = True
        layer_kwargs["seq_length"] = args.seq_length
    return layer_args, layer_kwargs


def parse_fsdp_args():
    parser = argparse.ArgumentParser(
        description="Run Transformer Engine modules with the "
        + "torch.distributed.fsdp.FullyShardedDataParallel strategy."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print out information from all GPUs instead of only the root GPU-0.",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=6, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=8192, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=64, help="Number of attention heads."
    )
    parser.add_argument("--num-kv", type=int, default=8, help="Number of KV heads.")
    parser.add_argument(
        "-d",
        "--head-dim",
        type=int,
        default=128,
        help="Dimension of each attention head (number of KV channels).",
    )
    parser.add_argument(
        "-i", "--num-iters", type=int, default=26, help="Number of dummy 'training' iterations."
    )
    parser.add_argument(
        "--ffn_hidden_size",
        type=int,
        default=28672,
        help="Dimension of FFN/MLP"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128256,
        help="Dimension of vocabulary size.",
    )  
    parser.add_argument(
        "-k",
        "--num-layers",
        type=int,
        default=80,
        help="Number of modules chained together with nn.Sequential.",
    )
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=2,
        help="Number of modules chained together with nn.Sequential.",
    )
    parser.add_argument(
        "--warm-up-steps",
        type=int,
        default=3,
        help="Number of modules chained together with nn.Sequential.",
    )
    parser.add_argument(
        "--layer-type",
        type=te_layer,
        default=te.TransformerLayer,
        choices=list(te_layer_map.values()),
        help="TE module type used to construct the test model.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="PyTorch RNG seed.")
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable memory profiling via torch.profiler.profile().",
    )
    parser.add_argument(
        "--profile-name", type=str, default=None, help="File path for memory profiling."
    )
    parser.add_argument(
        "--checkpoint-layer",
        type=te_layer,
        default=te.TransformerLayer,
        help="Recompute activations of the selected layer during the backward "
        + "pass instead of saving.",
    )
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        default=False,
        help="Disables the te.fp8_autocast() context.",
    )
    parser.add_argument(
        "--no-defer-init",
        action="store_true",
        help="Defer module parameter initialization until after FSDP sharding.",
    )
    parser.add_argument(
        "--no-te-fsdp",
        action="store_true",
        help="Disable sharding of intermediate/activation tensors in TE modules.",
    )
    parser.add_argument(
        "--dtype",
        type=torch_dtype,
        default=torch.bfloat16,
        help="Data type for input tensor and Transformer Engine module parameters.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )
    parser.add_argument(
        "--tensorboard_dir", type=str, 
        default="trace",
        help="Enable profiling",
    )
    parser.add_argument(
        "--profile_step_start", type=int, 
        default=5,
        help="Enable profiling",
    )
    parser.add_argument(
        "--profile_step_end", type=int, 
        default=8,
        help="Enable profiling",
    )
    parser.add_argument(
        "--log_dir", type=str, 
        default='./output/',
        help="Enable profiling",
    )
    return parser.parse_args()


def dist_print(text, all_ranks=False, no_new_line=False):
    if LOCAL_RANK == 0 or all_ranks:
        end = "" if no_new_line else "\n"
        print(f"[GPU-{LOCAL_RANK}] " + text, end=end)


def train(args):
    # Initialize torch.distributed global process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    dist_print(f"WORLD_SIZE = {WORLD_SIZE}")
    torch.manual_seed(args.seed)

    # setup dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../checkpoint/llama3-8b")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = args.vocab_size
    train_loader = CustomDataLoader(batch_size=args.batch_size, max_length=args.seq_length, tokenizer=tokenizer)
    input_embed_layer = nn.Embedding(vocab_size, args.num_heads * args.head_dim, dtype=args.dtype, device="cuda" if args.no_defer_init else "meta")

    # Construct a simple homogeneous model (only one layer type) with NO PARALLELISM
    layer_args, layer_kwargs = get_layer_args(args)
    linear_args, linear_kwargs = get_layer_args(args, layer_type=te.Linear)

    if args.num_layers > 1:
        # te_layer_list = []
        te_layer_list = [input_embed_layer]
        for i in range(args.num_layers):
            te_layer_list.append(te.TransformerLayer(
                            hidden_size=args.num_heads * args.head_dim,
                            num_attention_heads=args.num_heads,
                            num_gqa_groups=args.num_heads//args.num_kv,
                            attn_input_format='bshd',
                            attention_dropout=0.0,
                            normalization='RMSNorm',
                            layernorm_epsilon=1e-05,
                            ffn_hidden_size=args.ffn_hidden_size,
                            bias=False,
                            activation='swiglu',
                            hidden_dropout=0.0,
                            **layer_kwargs)
            )
        
        del linear_kwargs['bias']
        te_layer_list.append(te.LayerNormLinear(
            args.num_heads * args.head_dim, vocab_size, bias=False,
            normalization='RMSNorm', eps=1e-05, **linear_kwargs
        ))
        te_model = nn.Sequential(*te_layer_list)
    else:
        # Single layer model
        te_model = args.layer_type(*layer_args, **layer_kwargs)

    flops_per_iter = estimate_flops_per_token(args)

    # Print out allocated device memory before the model parameters are sharded by FSDP
    pre_mem_use = torch.cuda.memory_allocated(device=f"cuda:{LOCAL_RANK}") * 1e-6
    dist_print(f"Pre-FSDP memory use = {pre_mem_use}MiB")

    # Wrap the model with FSDP
    # NOTE: The TE model itself has no inherent parallelism. FSDP shards model parameters and
    #       controls all communication.
    all_gpus = dist.new_group(backend="nccl")
    fsdp_wrap_policy = always_wrap_policy
    if args.layer_type == te.TransformerLayer:
        # NOTE: FSDP causes illegal memory access without this special policy for Transformers
        fsdp_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={te.TransformerLayer}
        )
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    te_model = FullyShardedDataParallel(
        te_model,
        process_group=all_gpus,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=args.dtype,
            reduce_dtype=torch.float32,
        ),
        auto_wrap_policy=fsdp_wrap_policy,
    )

    if args.checkpoint_layer is not None:
        # Recompute the activations of the selected layer during the backward pass instead of
        # saving them during the forward pass
        apply_fsdp_checkpointing(te_model, blocks=args.checkpoint_layer)
    elif not args.no_te_fsdp:
        # Prepare TE modules to shard internal buffers that FSDP cannot shard on its own
        prepare_te_modules_for_fsdp(te_model)

    # Print out allocated device memory after the model parameters are sharded
    post_mem_use = torch.cuda.memory_allocated(device=f"cuda:{LOCAL_RANK}") * 1e-6
    dist_print(f"Post-FSDP memory use = {post_mem_use}MiB")
    # dist_print(f"FSDP-Wrapped + Checkpointed TE Model:\n{te_model}")

    # Fp8 setup for TE
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Optimizer must be created after the model is wrapped in FSDP and the parameters are sharded
    optim = torch.optim.SGD(te_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
    ce_loss_func = torch.nn.CrossEntropyLoss()
    iter_num = 0 
    if args.profile:
        prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start-1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end-args.profile_step_start,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
        record_shapes=True,
        with_stack=True)
        prof.start()

    if LOCAL_RANK == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if LOCAL_RANK == 0:
        last_time = time.time()

    iter_times = []
    for batch in train_loader.get_dataloader():
        if args.profile:
            prof.step()
        if iter_num==3:
            # Profile memory use
            if args.profile_memory:
                torch.cuda.memory._record_memory_history(max_entries=100000)
            else:
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
        input_tokens = batch['input_ids'].cuda()
        label = batch['labels'].transpose(0,1).cuda()

        # fp8_autocast needs to be given the FSDP process group for amax reductions
        with te.fp8_autocast(enabled=not args.no_fp8, fp8_recipe=fp8_recipe, fp8_group=all_gpus):
            y = te_model(input_tokens)

        # calculate gradient and take training step outside the fp8_autocast context
        loss = ce_loss_func(y.reshape(-1, vocab_size), label.reshape(-1))

        loss.backward()
        loss_value_reduce = all_reduce_mean(loss.item())
        if LOCAL_RANK == 0 and log_writer is not None:
            log_writer.add_scalar('train_loss', loss_value_reduce, iter_num)

        if iter_num>0 and iter_num % args.grad_acc_steps == 0:
            optim.step()
            optim.zero_grad(set_to_none=True)

            if LOCAL_RANK == 0:
                current_time = time.time()
                iter_time = (current_time-last_time) / args.grad_acc_steps
                token_per_sec = (args.batch_size * args.seq_length)/iter_time
                print(f"Step: {iter_num}; TFLOP/s: {flops_per_iter/iter_time/1e12}; iteration time: {iter_time}; token per second: {token_per_sec}")
                iter_times.append(iter_time)
                last_time = current_time

        if args.profile and iter_num == args.profile_step_end:
            prof.stop()

        if iter_num>=args.num_iters-1:
            break
        iter_num += 1

    if log_writer is not None:
        log_writer.flush()

    if args.profile_memory:
        torch.cuda.memory._dump_snapshot(f"gpu{LOCAL_RANK}_{args.profile_name}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    else:
        end.record()
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        train_time = start.elapsed_time(end) / 1000.0
        dist_print(f"Training Time: {train_time}s")
        dist_print(f"Avg. Iter. Time: {train_time / (args.num_iters-3)}s")
        dist_print(f"Peak Memory Use: {peak_mem * 1e-6}MBs")

    if LOCAL_RANK == 0:
        iter_times = np.array(iter_times)[2:-2] # remove the first two and last two steps
        avg_iter_time = np.mean(iter_times)
        print("Avg token per second:",(args.batch_size * args.seq_length)/avg_iter_time)
        print("Avg iter time:",avg_iter_time)
        print("TFLOP per iteration:",flops_per_iter/1e12)
        print("Avg TFLOP/s,",flops_per_iter/avg_iter_time/1e12)
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}") * 1e-6
        print(f"Peak memory use = {peak_memory}MB")

# Run with:
#   torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 6
if __name__ == "__main__":
    args = parse_fsdp_args()
    date_output = subprocess.check_output(['date', '+%Y-%m-%d_%H-%M-%S']).decode('utf-8').strip()
    args.log_dir = os.path.join(args.log_dir, date_output)
    train(args)


