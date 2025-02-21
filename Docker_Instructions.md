# ROCm PyTorch Training Container (v25.2)
Unified PyTorch base container optimized for training with ROCm. 

## Overview

PyTorch is an open-source machine learning framework that is widely used for model training with GPU-optimized components for transformer-based models. 

The ROCm PyTorch Training Docker `rocm/pytorch-training:v25.3` container, available through AMD Infinity Hub, provides prebuilt, optimized environment for fine-tuning, pre-training a model on the AMD Instinct™ MI300X and MI325X accelerator. This ROCm PyTorch Docker includes the following components:

| Software component  | Version            |
|---------------------|--------------------|
| ROCm               | 6.3.0              |
| Python            | 3.10               |
| PyTorch           | 2.7.0a0+git637433   |
| Transformer Engine | 1.11               |
| Flash Attention   | 2.6.3              |
| hipBLASLt         | latest          |
| Triton            | 3.1                 |

This guide provides steps for validating the training performance of the AMD Instinct ™ MI300X / MI325X accelerator-based server platform using multiple Docker ® benchmarking containers provided by AMD. 
Standard benchmark tests provide consistent and reproducible results, allowing fair and accurate comparisons between different systems. These provide a common reference for evaluating performance test outcomes.

There is a Docker container provided via AMD Infinity Hub, which covers the following training performance benchmarks:

1)	FLUX.1
2)	Llama 3.1 70B  (BF16)
3)	Llama 3.1 8B  (BF16 and FP8)
4)	Mistral-7B (BF16 and FP8)

Please note that some models, such as Llama 3, require an external license agreement through a third party (e.g. Meta).

This container should not be expected to provide generalized performance across all training workloads. Users should expect the container perform in the model configurations described below, but other configurations and run conditions are not validated by AMD. 

## Test Platform Configuration
The test procedures presented in this document utilize the containers provided by AMD and corresponding scripts to ensure accurate reproduction of expected performance.
The benchmarks in this document have been tested across a variety of platforms, with varying CPU models and system configurations. Application performance can exhibit variations due to platform differences.

This document represents data validated on a variety of these systems, predominantly validated against: 
• Ubuntu ® 22.04.5 LTS 
• Linux® kernel 5.15.0-122-generic 
• AMD ROCm™ software 6.3.0

Note: Keeping systems updated to the latest AMD ROCm release is recommended. The containerized benchmarks in this guide include a specific AMD ROCm version in the container against which they were optimized. Newer releases of software from AMD can deliver different performance. Server manufacturers may vary configurations, yielding different results. Performance may vary based on the use of the latest drivers and optimizations.

### Docker container setup
Each of the performance validation benchmarks in this document is containerized in a Docker container hosted by AMD. Access to these containers is available through [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html). 

#### Pull MI300X Training Docker image
```
docker pull rocm/pytorch-training:v25.2      
```
#### Run the container 
```
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name training_env  rocm/pytorch-training:v25.2
```
## Reference Performance 
This section provides detailed instructions for replicating the benchmarks. Each test described in this chapter has its own unique set of configurations and instructions to be followed in order to reproduce the results. For convenience, benchmarking scripts are included in the container to set up appropriate environment variables and launch the workloads. The purpose of this chapter is to detail the specific commands to reproduce the benchmarks, provide example output, and include any special considerations for running the benchmarks. 
### MI300X platform performance results
| **Models**                | **Precision** | **Batch Size** | **Sequence Length** | **TFLOPS/s/GPU**     |
|---------------------------|---------------|---------------:|--------------------:|----------------------|
| **Llama 3.1 70B with FSDP** | BF16         | 4             | 8192               | 426.79               |
| **Llama 3.1 8B with FSDP**  | BF16         | 3             | 8192               | 542.94               |
| **Llama 3.1 8B with FSDP**  | FP8          | 3             | 8192               | 737.40               |
| **Llama 3.1 8B with FSDP**  | BF16         | 6             | 4096               | 523.79               |
| **Llama 3.1 8B with FSDP**  | FP8          | 6             | 4096               | 735.44               |
| **Mistral 7B with FSDP**    | BF16         | 3             | 8192               | 483.17               |
| **Mistral 7B with FSDP**    | FP8          | 4             | 8192               | 723.30               |
| **FLUX**                    | BF16         | 10            | -                  | 4.51 (FPS/GPU)*      |

<sub>*Note: FLUX performance is measured in FPS/GPU rather than TFLOPS/s/GPU.</sub>

### MI325X platform performance results
| **Models**                 | **Precision** | **Batch Size** | **Sequence Length** | **TFLOPS/s/GPU** |
|----------------------------|---------------|---------------:|--------------------:|-----------------:|
| Llama 3.1 70B with FSDP    | BF16          | 7             | 8192               | 526.13          |
| Llama 3.1 8B with FSDP     | BF16          | 3             | 8192               | 643.01          |
| Llama 3.1 8B with FSDP     | FP8           | 5             | 8192               | 893.68          |
| Llama 3.1 8B with FSDP     | BF16          | 8             | 4096               | 625.96          |
| Llama 3.1 8B with FSDP     | FP8           | 10            | 4096               | 894.98          |
| Mistral 7B with FSDP       | BF16          | 5             | 8192               | 590.23          |
| Mistral 7B with FSDP       | FP8           | 6             | 8192               | 860.39          |

### FLUX.1
Users can test image generation training throughput from the FLUX.1-dev model with the best batch size before the runs go out of memory. FlashAttention version 2.6.3 is enabled in all tests.

The throughput benchmark test was run with 8 AMD Instinct™ MI300X or MI325X accelerators running the AMD’s [FluxBenchmark](https://github.com/ROCm/FluxBenchmark) repository based on PyTorch and Hugging Face’s accelerate library.
This document focuses on GPU-normalized Training Throughput (frame per second per GPU).

#### Test Method
FLUX.1-dev is a 12-billion-parameter rectified flow transformer developed by Black Forest Labs. In this test, the model was fine-tuned from its openly available pretrained weights on pseudo-camera-10k dataset with BF16 precision, with best batch before going out of memory on in the MI300X or MI325X accelerators. 
 
Accelerate is a library developed by Hugging Face and is a tool for simplifying and optimization the process of training and deploying AI models, and it is built on top of PyTorch library. 

We utilized the containers provided by AMD and tuning some environmental parameters and requirement installation.

#### Test Procedures
Create the benchmark FluxBenchmark (https://github.com/ROCm/FluxBenchmark), which aligns with original Black Forest Labs' implementation (https://github.com/black-forest-labs/flux). 

Make sure you are in the Docker container “training_env” created in Docker container setup step. If not, start the Docker container and execute it.
```
#Optioinal, execute the training_env container
docker start training_env
docker exec -it training_env bash
```

Enter FluxBenchmak
```
cd FluxBenchmark 
```

Set up environment
```
pip3 install --no-cache-dir --upgrade pip packaging
pip3 install --no-cache-dir -r requirements.txt
export HF_HOME=/workspace/huggingface
export ROCBLAS_USE_HIPBLASLT=1
export DISABLE_ADDMM_CUDA_LT=0
export HIP_FORCE_DEV_KERNARG=1
export TORCH_NCCL_HIGH_PRIORITY=0
export GPU_MAX_HW_QUEUES=8
```
Download the necessary assets
```
make download_assets
```
Launch the benchmarking script
```
python launcher.py
```
Note: Launcher runs sweeps across different configs. Edit config/minimal_launcher_config.yaml inside of the Docker and add/remove the parameters as needed.

#### References
* AMD FluxBenchmark GitHub: https://github.com/ROCm/FluxBenchmark
* HuggingFace model card for FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev 
* HuggingFace dataset card for pseudo-camera-10k: https://huggingface.co/datasets/bghira/pseudo-camera-10k/ 

### Llama 3.1 70B 

Llama-3.1-70B is a generative text model with 70 billion parameters based on the Llama-3.1 architecture. PyTorch is a framework for large language model (LLM) training with GPU-optimized components for transformer-based models. 

Users can test text generation training throughput from the Llama-3.1-70B model with a maximum sequence length of 8192 tokens and batch size of 4 on MI300X system and batch size of 7 on MI325X system. 
 
The test was run on the system, which has 8 AMD Instinct™ MI300X or MI325X accelerators. This test focuses on GPU-normalized Training Throughput (TFLOPs per second per GPU). 

#### Test Method
In this test, the model was pre-trained on the c4_test dataset using a maximum sequence length of 8192 tokens. Model training was executed on a system with 8 MI300X/MI325X accelerators. The test used FSDP2, tensor parallelism of 1, pipeline parallelism of 1, and data parallelism of 1. The largest micro-batch size that fits in GPU memory was used in order to maximize training throughput. 

Note:
1.	This test modified the original training configuration for 70B models. The updated configuration has the following benefits: (1) It used the maximal batch size that can fit the MI300X accelerator, which means fully utilize its memory capacity; and (2) It switched from tensor parallelism from 8 to 1 and used the FSDP2 which can significantly reduce the communication bubbles during training. 
2.	This test modified the interface for torch.compile. The modified torch.compile provided the following benefits: (1) It removed some unnecessary memory consumption; and (2) It reduced overall computation latency for small operators. 

#### Test Procedures
Leverage the public code base [torchtitan](https://github.com/pytorch/torchtitan/tree/main), to automate data preparation, and execute benchmark runs. AMD made some minor changes to the original code base. Please refer to the README in the top-level directory of torchtitan for full details.  

Make sure you are in the Docker container “training_env” created in Docker container setup. If not, start the Docker container and execute it.
```
#optioinal, execute the training_env container
docker start training_env
docker exec -it training_env bash
```
Enter torchtitan directory
```
cd torchtitan  
```
Execute the test 
```
pip install -r requirements.txt
bash run_llama_train.sh 
```

#### References
ROCM/torchtitan GitHub: [torchtitan](https://github.com/ROCm/torchtitan)

### Llama 3.1 8B 
Llama-3.1-8B is a generative text model with 8 billion parameters based on the Llama-3.1 architecture. PyTorch is a framework for large language model (LLM) training with GPU-optimized components for transformer-based models. 

Users can test text generation training throughput from the Llama-3.1-8B model with a maximum sequence length up to 8192 tokens with various batch sizes. 
 
The test was run on the system, which has 8 AMD Instinct™ MI300X or MI325X accelerators. This test focuses on GPU-normalized Training Throughput (TFLOPs per second per GPU). 

#### Test Method
In this test, the model was pre-trained on the synthesized dataset using a maximum sequence length of 4096 tokens. Model training was executed on a system with 8 MI300X/MI325 accelerators with tensor parallelism of 1, pipeline parallelism of 1, and data parallelism of 8. The largest micro-batch size that fits in GPU memory was used in order to maximize training throughput.  

#### Test Procedures
We use the code base https://github.com/AMD-AIG-AIMA/pytorch-training-benchmark to automate data preparation, and execute benchmark runs. Please refer to the README for more details.

Make sure you are in the Docker container “training_env” created in Docker container setup. If not, start the Docker container and execute it.
```
#optioinal, execute the training_env container
docker start training_env
docker exec -it training_env bash
```

Enter Training directory
```
cd /workspace/pytorch-training-benchmark  
```
Execute the test run with BF16 precision 
```
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="0.0.0.0" --master_port="12234" ./train_fsdp.py configs/llama-3.1-8b-4k.json llama --batch_size 3 |& tee -a ./llama_bf16.log 
```
Execute the test run with FP8 precision 
```
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="0.0.0.0" --master_port="12234" ./train_fsdp.py configs/llama-3.1-8b-4k.json llama --batch_size 3 --enable_fp8 True |& tee -a ./llama_fp8.log
```

### Mistral 7B 

Users can test text generation training throughput from Mistral-7B with FP8 precision and  with BF16 precision.

The test was run on the system, which has 8 AMD Instinct™ MI300X or MI325X accelerators. This test focuses on GPU-normalized Training Throughput (TFLOPs per second per GPU). 

#### Test Method
PyTorch is a framework for large language model (LLM) training with GPU-optimized components for transformer-based models.  

Models in this section are pre-trained on a synthesized dataset with the maximum sequence length as stated in the table for the different variants. Model training was executed on a system with 8 MI300X or MI325X accelerators with FSDP strategy for Mistral 7B, the largest micro-batch size that fits in GPU memory was used in order to maximize training throughput.    

#### Test Procedures
Use the Pytorch to run the benchmarks available at(https://github.com/AMD-AIG-AIMA/pytorch-training-benchmark)
  
Make sure you are in the Docker container “training_env” created in Docker container setup. If not, start the Docker container and execute it.
```
# Optional, execute the training_env container
docker start training_env
docker exec -it training_env bash
```

Enter Training directory
```
cd /workspace/pytorch-training-benchmark
``` 
Run benchmarks with BF16 precision
```
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="0.0.0.0" --master_port="12234" ./train_fsdp.py configs/mistral-7b-v0.1.json mistral --batch_size 3 |& tee -a ./mistral_bf16.log 
```
Run benchmarks with FP8 precision
```
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="0.0.0.0" --master_port="12234" ./train_fsdp.py configs/mistral-7b-v0.1.json mistral --batch_size 4 --enable_fp8 True |& tee -a ./mistral_fp8.log
```

## Reference System Configurations
The benchmarks in this document have been tested across a variety of platforms, with various CPU models and system configurations. Key system configuration details are described below.  
#### MI300X-based system
| **Item**                | **Details**                                                                                                                                                                                 |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Summary Description** | Supermicro AS -8125GS-TNMR2 with 2x AMD EPYC 9654 96-Core Processor, 8x AMD Instinct MI300X (192GiB, 750W) accelerators, Ubuntu 22.04, ROCm 6.3, DDR5 Memory, and PCIe 5.0 support           |
| **System Model**        | Supermicro AS -8125GS-TNMR2                                                                                                                                                                 |
| **System BIOS**         | Ver 3.2, SMT disabled                                                                                                                                                                        |
| **CPU**                 | 2x AMD EPYC 9654 96-Core Processor (2 sockets, 96 cores per socket, 2 threads per core)                                                                                                      |
| **NUMA Config**         | 1 NUMA node per socket                                                                                                                                                                       |
| **Memory**              | 2304 GB (24 DIMMs, 4800 MT/s, 96 GB/DIMM)                                                                                                                                                    |
| **Disk**                | - Root drive: 2x 960GB Samsung MZ1L2960HCJR-00A07 NVMe SSDs<br/>- Data drives: 4x 3.84TB Samsung MZQL23T8HCLS-00A07 NVMe SSDs                                                                |
| **GPU**                 | 8x MI300X, 192GB HBM3e, 750W                                                                                                                                                                 |
| **Host OS**             | Ubuntu 22.04.5 LTS with Linux kernel 5.15.0-122-generic                                                                                                                                      |
| **Host GPU Driver**     | ROCm 6.3.0.60300-39~22.04 + amdgpu 6.10.5.60300-2084815.22.04                                                                                                                                |
#### MI325X-based system
| **Item**                | **Details**                                                                                                                                                          |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Summary Description** | Dell PowerEdge XE9680 with 2x Intel Xeon Platinum 8480+ Processors, 8x AMD Instinct MI325X (256GiB, 1000W) accelerators, Ubuntu 22.04, and a pre-release build of ROCm 6.3 |
| **System Model**        | Dell PowerEdge XE9680                                                                                                                                               |
| **CPU**                 | 2x Intel Xeon Platinum 8480+ Processors (2 sockets, 56 cores per socket, 2 threads per core)                                                                         |
| **NUMA Config**         | 1 NUMA node per socket                                                                                                                                               |
| **Memory**              | 4096 GB (32 DIMMs, 4400 MT/s, 128 GB/DIMM)                                                                                                                           |
| **Disk**                | - Root drive: 2x 480GB Dell EC NVMe ISE 7450 RI M.2 80<br/>- Data drives: 2x 1.6TB Dell Ent NVMe PM1735a MU                                                           |
| **GPU**                 | 8x MI325X, 256GB HBM3e, 1000W                                                                                                                                       |
| **Host OS**             | Ubuntu 22.04.5 LTS with Linux kernel 5.15.0-122-generic                                                                                                              |
| **Host GPU Driver**     | ROCm 6.3.0 (pre-release build 14701) + amdgpu 6.9.5                                                                                                                  |

# Acknowledgment
We acknowledge SemiAnalysis LLC, whose [benchmarking code](https://hub.docker.com/r/semianalysiswork/single-amd-vip-nov-25) served as the foundation for this setup.


