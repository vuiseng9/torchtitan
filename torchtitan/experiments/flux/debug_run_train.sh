#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./torchtitan/experiments/flux/run_train.sh

NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/flux/train_configs/debug_model.toml"}
DEBUG=${DEBUG:-1}
if [ $DEBUG == 1 ]; then
    DEBUG_FLAG="-m debugpy --listen 0.0.0.0:5678 --wait-for-client"
else
    DEBUG_FLAG=""
fi
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export PYTHONPATH=$(pwd)

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="127.0.0.1:29500" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
${DEBUG_FLAG} \
-m torchtitan.experiments.flux.train --job.config_file ${CONFIG_FILE} $overrides


# IMPORTANT: cd to the directory containing the train module, which is also where this script resides.
# ./debug_run_train.sh # single gpu via torch distributed, DEBUG is turned on by default

# attach debugger in vscode
        # {
        #     "name": "Python Debugger: Remote Attach",
        #     "type": "debugpy",
        #     "request": "attach",
        #     "connect": {
        #         "host": "localhost",
        #         "port": 5678
        #     },
        #     "pathMappings": [
        #         {
        #             "localRoot": "${workspaceFolder}",
        #             "remoteRoot": "${workspaceFolder}"
        #         }
        #     ],
        #     "justMyCode": false
        # },
