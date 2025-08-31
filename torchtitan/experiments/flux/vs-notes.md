Intent
1) use mlperf commit, particular torchtitan latest are still moving rapidly
2) reproduce with debug-ability
3) tiny training, full blown c12m later

[MLPerf Page](https://github.com/mlcommons/training/tree/master/text_to_image)

```bash
install-torch 128

# Non-recurring (how i create this branch)
#     git clone https://github.com/vuiseng9/torchtitan
#     cd torchtitan
#     git remote add upstream https://github.com/pytorch/torchtitan
#     git fetch upstream 6325268884be2da5b3a42556055ecd0018b8731e
#     git checkout FETCH_HEAD -b 250813-mlperf-commit-63252

git clone https://github.com/vuiseng9/torchtitan
cd torchtitan
git checkout 250813-mlperf-commit-63252
# pip install -e . # we dont really need to install
pip install -r requirements.txt

git clone https://github.com/NVIDIA/mlperf-common.git
pip install -e mlperf-common

git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging

cd torchtitan/torchtitan/experiments/flux/
pip install -r  requirements-flux.txt 

# Access to HF collaterals
huggingface-cli login

# for debug
pip install debugpy
```

> Note that the design of experiments and most scripts are meant to be run from root torchtitan/ folder.

`Makefile` has been provided in root dir for the following.

### Dev/Debug
use `vscode/{launch,tasks}.json` in `<workspace>/.vscode`. Copy or Softlinks will do.

### Dataset Preparation
* COCO 2014 Validation Set. According to [mlperf-training](https://github.com/mlcommons/training/tree/master/text_to_image#coco-2014-subset)
    1. (non-recurring)`make dl-coco-2014-val` downloads coco-2014 validation dataset
    2. (non-recurring)`make coco-2014-val-wds` create the validation subset, resize to 256x256 and convert to webdataset
    3. (Reuse)`make dl-wds-coco-2014-val`

### Launch Single GPU debug
* F5 on "Attach torchrun"
