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
