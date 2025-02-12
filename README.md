# Training
This code is used for benchmarking Pytorch based pre-training on a synthesized dataset for a single node using [Torchrun](https://pytorch.org/docs/stable/elastic/run.html) utility.

```
torchrun [TORCHRUN_PARAMETERS] ./train_fsdp.py <model_config_file> <model_type>[llama/mistral] --batch-size[DEFAULT=1]
```

## Parameters
* **num_iteration**:
The default setting will run the pre-training for 128 steps and use the later 64 steps to calculate the thoughput. You can change this by change num_iteration.
* **enable_fp8**:
This will enable fp8 for training. The default setting is False.
* **enable_compile**:
This will enable/disable pytorch compiling. Compiling is disabled by default when using fp8 and enabled by default for bf16.
* **batch_size**:
This will set the batch size used for pretraining. Default is set to 1. 

Listed below are some example run commands for the model benchmarked in this repository using FSDP sharding strategy.

## Run commands for Mistral training with 8k sequence length
### MI300
### BF16 Precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/mistral-7b-v0.1.json mistral --batch_size 3  |& tee -a ./mistral_bf16.log
```
### FP8 Precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/mistral-7b-v0.1.json mistral  --batch_size 4 --enable_fp8 True |& tee -a  ./mistral_fp8.log
```
### MI325
#### BF16 Precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/mistral-7b-v0.1.json mistral --batch_size 5  |& tee -a ./mistral_bf16.log
```
#### FP8 Precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/mistral-7b-v0.1.json mistral  --batch_size 6 --enable_fp8 True |& tee -a  ./mistral_fp8.log
```
## Run commands for Llama3.1-8B training with 4k sequence length
### MI300
### BF16 precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 6 |& tee -a ./llama_bf16.log
```
### FP8 Precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 6 --enable_fp8 True |& tee -a ./llama_fp8.log
```
### MI325
### BF16 precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 8 |& tee -a ./llama_bf16.log
```
### FP8 Precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 10 --enable_fp8 True |& tee -a ./llama_fp8.log
```
## Run commands for Llama3.1-8B training with 8k sequence length
### MI300
### BF16 precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-8k.json llama --batch_size 3 |& tee -a ./llama_bf16.log
```
### FP8 Precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-8k.json llama --batch_size 3 --enable_fp8 True |& tee -a ./llama_fp8.log
```
### MI325
### BF16 precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-8k.json llama --batch_size 3 |& tee -a ./llama_bf16.log
```
### FP8 Precision
```
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-8k.json llama --batch_size 5 --enable_fp8 True |& tee -a ./llama_fp8.log
```
# Finetuning
This section describes finetuning llama-3.1-70b using wikitext dataset on a single node using [Torchtune](https://github.com/AMD-AIG-AIMA/torchtune-private) utility.

### Environment setup

This installs torch 2.7.0a0+git6374332 and the torch.compile works fine within the docker.

```bash
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged    -v  $HOME/.ssh:/root/.ssh  -v /home/amd:/home/amd --shm-size 128G --name YOUR_NAME_HERE rocm/pytorch-training-private:20250207
pip3 install torchao --index-url https://download.pytorch.org/whl/nightly/rocm6.3

# This is the main branch
git clone https://github.com/AMD-AIG-AIMA/torchtune.git

cd torchtune

pip install -e .

# if you don't have access to this model, see the section below for an alternative source.
# but for formal testing we should use the correct model, not the unofficial mirror.
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir ./models/Llama-3.1-70B-Instruct --exclude 'original/*.pth'

# To download the wikitext dataset, go to "pytorch-training-benchmark" directory and do (train and test splits will be saved):
python dataset.py

# If any error downloading the data, do:
pip install datasets

# For full finetuning, go to "torchtune" directory and do:
# Copy both the 'wikitext_finetune.sh' and 'llama_3_1_70b_full_finetune_recipe.yaml' into the torchtune directory
cp -r ../pytorch-training-benchmark/wikitext_finetune.sh .
cp -r ../pytorch-training-benchmark/llama_3_1_70b_full_finetune_recipe.yaml .

# For LORA finetuning, go to "torchtune" directory and do:
# Copy both the 'wikitext_lora_finetune.sh' and 'llama_3_1_70b_lora_finetune_recipe.yaml' into the torchtune directory
cp -r ../pytorch-training-benchmark/wikitext_lora_finetune.sh .
cp -r ../pytorch-training-benchmark/llama_3_1_70b_lora_finetune_recipe.yaml .
```

### Full Finetuning Testing Command
The script `wikitext_finetune.sh` runs the finetuning test on `llama-3.1-70b` model with a wikitext dataset on top of the docker. Remove `MAX_STEPS=30` if you want to run for 1 complete epoch.
```
MODEL_DIR=./models/Llama-3.1-70B-Instruct COMPILE=True CPU_OFFLOAD=False PACKED=False SEQ_LEN=null ACTIVATION_CHECKPOINTING=True TUNE_ENV=True MBS=64 GAS=1 EPOCHS=1 SEED=42 VALIDATE=True MAX_STEPS=30 bash wikitext_finetune.sh
```

### LORA Finetuning Testing Command
The script `wikitext_finetune.sh` runs the finetuning test on `llama-3.1-70b` model with a wikitext dataset on top of the docker. Remove `MAX_STEPS=30` if you want to run for 1 complete epoch.
```
MODEL_DIR=./models/Llama-3.1-70B-Instruct COMPILE=True CPU_OFFLOAD=False PACKED=False SEQ_LEN=null ACTIVATION_CHECKPOINTING=True TUNE_ENV=True MBS=64 GAS=1 EPOCHS=1 SEED=42 VALIDATE=True MAX_STEPS=30 bash wikitext_lora_finetune.sh
```

### Performance Result (Full Finetuning)
Result for `MAX_STEPS=30` on a single node (8 GPUs) - AMD Instinct MI300X:TW044
```
Max memory alloc: 137.2001576423645
Average tokens/s/gpu: 92.0694
Unmasked tokens/s/gpu:  145.143
```

### Performance Result (LORA Finetuning)
Result for `MAX_STEPS=30` on a single node (8 GPUs) - AMD Instinct MI300X:TW044
```
Max memory alloc: 117.79637384414673
Average tokens/s/gpu: 65.7681
Unmasked tokens/s/gpu:  169.299
```

# Environment setup
We acknowledge SemiAnalysis LLC, whose [benchmarking code](https://hub.docker.com/r/semianalysiswork/single-amd-vip-nov-25) served as the foundation for this setup.
