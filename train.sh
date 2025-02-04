# cd /home/guihonli/project/training/
# pip install fire transformers accelerate setuptools===69.5.1 datasets numpy==1.26.4
# huggingface-cli login --token YOUR_HF_KEY

# torchrun --nproc_per_node=8  --master_port="12234" ./train_fsdp.py \
#     configs/llama-3.1-8b-4k.json llama --batch_size 4 --dummy_data False --num_iteration 3000 |& tee -a ./llama_bf16.log

# torchrun --nproc_per_node=8  --master_port="12235" ./train_fsdp.py \
#     configs/llama-3.1-8b-4k.json llama --batch_size 4 --enable_fp8 True --dummy_data False  --num_iteration 3000  |& tee -a ./llama_fp8.log


# torchrun --nproc_per_node=8  --master_port="12236" ./train_fsdp.py \
#     configs/mistral-7b-v0.1.json mistral --batch_size 2  --dummy_data False  --num_iteration 3000 |& tee -a ./mistral_bf16.log


# torchrun --nproc_per_node=8  --master_port="12237" ./train_fsdp.py \
#     configs/mistral-7b-v0.1.json mistral --batch_size 2 --enable_fp8 True  --dummy_data False  --num_iteration 3000 |& tee -a ./mistral_fp8.log

# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 2  |& tee -a bs2_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 3  |& tee -a bs3_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 2  |& tee -a bs2_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 3  |& tee -a bs3_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 2  |& tee -a bs2_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 3  |& tee -a bs3_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 2  |& tee -a bs2_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 3  |& tee -a bs3_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 2  |& tee -a bs2_fsdp_fp8.log
# torchrun --nnodes=1 --nproc-per-node=8 fsdp_fp8.py -b 3  |& tee -a bs3_fsdp_fp8.log

torchrun  --nproc_per_node=8  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 4 |& tee -a ./llama_bf16_BSDP_CUDNNATTN_2_3.log

torchrun  --nproc_per_node=8  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 4 |& tee -a ./llama_bf16_BSDP_CUDNNATTN_2_3.log

#torchrun  --nproc_per_node=8  ./train_fsdp.py \
#    configs/llama-3.1-8b-4k.json llama --batch_size 4 |& tee -a ./llama_bf16_baseline_2_3.log

#torchrun  --nproc_per_node=8  ./train_fsdp.py \
#    configs/llama-3.1-8b-4k.json llama --batch_size 4 |& tee -a ./llama_bf16_baseline_2_3.log

#torchrun  --nproc_per_node=8  ./train_fsdp.py \
#    configs/llama-3.1-8b-4k.json llama --batch_size 4 |& tee -a ./llama_bf16_baseline_2_3.log

torchrun  --nproc_per_node=8  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 5 |& tee -a ./llama_bf16_BSDP_CUDNNATTN_2_3_bs5.log

#/home/guihonli/project/training/llama_bf16_gl_enable_sdpa_gqa.log



