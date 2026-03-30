# https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/
# official guide: https://github.com/nebius/ml-cookbook/blob/main/pytorch-dsv3-mxfp8/README.md
# official requires soperator, our attempt below is to bypass and run on local node only.

dont use because of incompatible torch, need some new apis
nvcr.io/nvidia/pytorch:25.06-py3

verda image with cuda12.8 but driver 13.0

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0

install-torch 128
torch: 2.11.0+cu128
cuda : 12.8
cudnn: 91900
nccl: (2, 28, 9)
nvshmem 3.4.5

      pip install ninja

      sudo apt-get update
      sudo apt-get install -y rdma-core libibverbs-dev librdmacm-dev
      sudo apt-get install -y build-essential

# Build DeepEP
    ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

    # pip install nvidia-nvshmem-cu12 <torch will install>

    # find your own lib path 
    cd $(get-site-packages-path)/nvidia/nvshmem/lib
    ln -s libnvshmem_host.so.3 libnvshmem_host.so

    # commit: 29d31c09
    git clone https://github.com/vuiseng9/deepseek-ep -b main
    cd deepseek-ep
    NVSHMEM_DIR=$(get-site-packages-path)/nvidia/nvshmem TORCH_CUDA_ARCH_LIST=10.0 python setup.py install
    python tests/test_intranode.py --num-processes 2

# why we choose 0.16? the release date is closed to the torchtitan Nightly used by nebius
# pip install torchao==0.16.0 --index-url https://download.pytorch.org/whl/cu128

# torchtitan setup
# This branch is created in the following way (capture here for reference, no need to execute, just pip install -e .)
    git clone https://github.com/pytorch/torchtitan.git
    cd torchtitan && git checkout 1a36996b8dffb464340c280d3f0a8139ca4c36b6  # Nightly accessed 2026-02-03
    pip install -e .


pip install wandb transformers
hf auth login
wandb login

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/deepseek-moe-16b-base', local_dir='assets/hf/deepseek-moe-16b-base', revision='521d2bc4fb69a3f3ae565310fcc3b65f97af2580', allow_patterns=['tokenizer*', 'special_tokens*'])
"
snapshot_download('deepseek-ai/DeepSeek-V3.1-Base', local_dir='assets/hf/DeepSeek-V3.1-Base', revision='d3d4eafdc470de44bbf6f0a74f852eb522357be8', allow_patterns=['tokenizer*', 'special_tokens*'])
"

source train_16b.sh

c-diff ori_train_16b.sh train_16b.sh