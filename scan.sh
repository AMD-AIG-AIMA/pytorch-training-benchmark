# torchrun  --nproc_per_node=8  ./train_fsdp.py \
#     configs/llama-3.1-8b-4k.json llama --batch_size 2 |& tee -a ./llama_bf16_cudnn_att.log

# torchrun  --nproc_per_node=8  ./train_fsdp.py \
#     configs/llama-3.1-8b-4k.json llama --batch_size 2 |& tee -a ./llama_bf16_cudnn_att.log

# torchrun  --nproc_per_node=8  ./train_fsdp.py \
#     configs/llama-3.1-8b-4k.json llama --batch_size 2 |& tee -a ./llama_bf16_cudnn_att.log

# torchrun  --nproc_per_node=8  ./train_fsdp.py \
#     configs/llama-3.1-8b-4k.json llama --batch_size 2 |& tee -a ./llama_bf16_cudnn_att.log

# torchrun  --nproc_per_node=8  ./train_fsdp.py \
#     configs/llama-3.1-8b-4k.json llama --batch_size 2 |& tee -a ./llama_bf16_cudnn_att.log


torchrun  --nproc_per_node=8  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 4 |& tee -a ./llama_bf16_fav3_bs4.log

torchrun  --nproc_per_node=8  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 5 |& tee -a ./llama_bf16_fav3_bs5.log

torchrun  --nproc_per_node=8  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 5 |& tee -a ./llama_bf16_fav3_bs5.log

torchrun  --nproc_per_node=8  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 5 |& tee -a ./llama_bf16_fav3_bs5.log

torchrun  --nproc_per_node=8  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 5 |& tee -a ./llama_bf16_fav3_bs5.log






