# training

# Run commands for Mistral training
## BF16 Precision
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/mistral-7b-v0.1.json mistral --batch_size 3  |& tee -a ./mistral_bf16.log
## FP8 Precision
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/mistral-7b-v0.1.json mistral  --batch_size 4 --enable_fp8 True |& tee -a  ./mistral_fp8.log

# Run commands for Llama3.1-8B training
## BF16 precision
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b.json llama --batch_size 3 |& tee -a ./llama_bf16.log
## FP8 Precision
torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b.json llama --batch_size 3 --enable_fp8 True |& tee -a ./llama_fp8.log