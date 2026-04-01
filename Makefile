ngpu ?= $(shell nvidia-smi -L | wc -l)
dtype ?= float
torchrun_intra = torchrun --standalone --nproc-per-node
MASTER_ADDR ?= 10.13.113.101
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)
DBG ?= 0

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

dl-llama3.1-tokenizer:
	# do login hf
	python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer 

dryrun-train-llama3.1:
	# COMM_MODE is empty and will go through normal training with torchrun in bash script
	DBG_ATTACH=$(DBG) NGPU=$(ngpu) MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh

dbg-dryrun-train-llama3.1:
	$(MAKE) dryrun-train-llama3.1 DBG=1