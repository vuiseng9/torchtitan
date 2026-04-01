ngpu ?= $(shell nvidia-smi -L | wc -l)
# dtype ?= float
# torchrun_intra = torchrun --standalone --nproc-per-node
MASTER_ADDR ?= localhost
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)
DBG ?= 0

# ootb setup
# 1. optional rebase, add-upstream and rebase-upstream
# 2. install-from-source
# 3. install-aux for debugpy
# 4. login hf
# 5. dl-llama3.1-tokenizer for tokenizer assets 
# 6. dryrun-train-llama3.1

add-upstream:
	git remote add upstream https://github.com/pytorch/torchtitan

rebase-upstream:
	git fetch upstream
	git rebase upstream/main

install-from-source:
	pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
	pip install -r requirements.txt
	pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu

install-aux:
	pip install debugpy

install-ao-nightly:
	pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128

dl-llama3.1-tokenizer:
	# do login hf
	python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer 

_train-llama3:
	# COMM_MODE is empty and will go through normal training with torchrun in bash script
	DBG_ATTACH=$(DBG) NGPU=$(ngpu) MODULE=llama3 CONFIG=$(MCFG) ./run_train.sh

# --- llama3 debug model ---

dryrun-llama3-dbgmdl:
	$(MAKE) _train-llama3 MCFG=llama3_debugmodel DBG=$(DBG)
	
emulate-f8-llama3-dbgmdl:
	$(MAKE) _train-llama3 MCFG=llama3_debugmodel_float8_emulate DBG=$(DBG)

# --- llama3.1-8b ---

train-llama3.1-8b:
	$(MAKE) _train-llama3 MCFG=llama3_8b DBG=$(DBG)

dbg-train-llama3.1-8b:
	$(MAKE) train-llama3.1-8b DBG=1

f8-train-llama3.1-8b:
	$(MAKE) _train-llama3 MCFG=llama3_8b_float8 DBG=$(DBG)

emulate-f8-llama3.1-8b:
	$(MAKE) _train-llama3 MCFG=llama3_8b_float8_emulate DBG=$(DBG)

h-train-llama3.1-8b:
	python -m torchtitan.train --module llama3 --config llama3_8b --help

# --metrics.enable_wandb