#!/usr/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to preprocess the tiny CC12M dataset (or any small dataset)
# This is a simplified version of run_preprocessing.sh optimized for small datasets.
#
# Usage:
#   NGPU=1 bash run_tiny_preprocessing.sh --training.dataset_path=/dataset/cc12m_tiny
#
# Environment variables:
#   NGPU: Number of GPUs to use (default: 1)
#   CONFIG_FILE: Config file to use (default: flux_schnell_mlperf.toml)
#
# The script will output preprocessed data to /dataset/cc12m_tiny_preprocessed by default

set -ex

# Default configuration
NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml"}
export HF_HUB_CACHE=/root/.cache/huggingface/hub/

# Default paths - can be overridden via command line
INPUT_PATH=${INPUT_PATH:-"/dataset/cc12m_tiny"}
OUTPUT_PATH=${OUTPUT_PATH:-"/dataset/cc12m_tiny_preprocessed"}
BATCH_SIZE=${BATCH_SIZE:-"32"}  # Smaller batch size for tiny dataset

# Collect overrides from command line
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

echo "=============================================="
echo "Preprocessing Tiny CC12M Dataset"
echo "=============================================="
echo "Input path: ${INPUT_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Number of GPUs: ${NGPU}"
echo "Batch size: ${BATCH_SIZE}"
echo "=============================================="

# Run preprocessing with the tiny dataset
# Key differences from full preprocessing:
# - Smaller batch size (32 vs 256)
# - Uses cc12m_tiny dataset name
# - Single GPU by default (can use more if available)
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.experiments.flux.scripts.preprocess_flux_dataset --job.config_file ${CONFIG_FILE} \
--experimental.custom_args_module torchtitan.experiments.flux.preprocessing_config \
--eval.dataset= \
--checkpoint.no_enable_checkpoint \
--training.batch_size ${BATCH_SIZE} \
--training.dataset=cc12m_tiny \
--training.dataset_path=${INPUT_PATH} \
--preprocessing.output_dataset_path=${OUTPUT_PATH} \
--parallelism.data_parallel_replicate_degree=${NGPU} \
--training.classifer_free_guidance_prob=0.0 \
--model.flavor=flux-debug $overrides

# Generate empty encodings (needed for classifier-free guidance)
echo "=============================================="
echo "Generating empty encodings..."
echo "=============================================="

# make empty encodings directory at same level as OUTPUT_PATH
ENCODINGDIR=$(dirname ${OUTPUT_PATH})/empty_encodings
mkdir -p ${ENCODINGDIR}
torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
-m torchtitan.experiments.flux.scripts.save_empty_encodings --job.config_file ${CONFIG_FILE} \
--experimental.custom_args_module torchtitan.experiments.flux.preprocessing_config \
--eval.dataset= \
--preprocessing.output_dataset_path=${ENCODINGDIR} \
--training.classifer_free_guidance_prob=0.0 \
--checkpoint.no_enable_checkpoint --training.batch_size 256 --training.dataset=dummy \
--model.flavor=flux-debug --encoder.empty_encodings_path=

echo "=============================================="
echo "Preprocessing complete!"
echo "=============================================="
echo "Preprocessed data saved to: ${OUTPUT_PATH}"
echo "Empty encodings saved to: ${ENCODINGDIR}"
echo ""
echo "To train with preprocessed data, use config:"
echo "  flux_schnell_mlperf_preprocessed.toml"
echo "And override dataset paths:"
echo "  --training.dataset_path=${OUTPUT_PATH}"
echo "=============================================="
