ngpu ?= $(shell nvidia-smi -L | wc -l)
dtype ?= float
torchrun_intra = torchrun --standalone --nproc-per-node
MASTER_ADDR ?= 10.13.113.101
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)
DBG ?= 0

get-dsv3-16b-tokenizer:
	python3 -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/deepseek-moe-16b-base', local_dir='assets/hf/deepseek-moe-16b-base', revision='521d2bc4fb69a3f3ae565310fcc3b65f97af2580', allow_patterns=['tokenizer*', 'special_tokens*'])"

dbg-intranode-dsv3-16b-bf16:
	$(MAKE) intranode-dsv3-16b-bf16 DBG=1

intranode-dsv3-16b-bf16:
	DBG_ATTACH=$(DBG) $(torchrun_intra) $(ngpu) \
	-m torchtitan.train \
	--job.config_file torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml \
	--metrics.log_freq=1 \
	--training.steps=1500 \
	--parallelism.data_parallel_replicate_degree=1 \
	--parallelism.data_parallel_shard_degree=-1 \
	--parallelism.expert_parallel_degree=2 \
	--parallelism.tensor_parallel_degree=1 \
	--parallelism.expert_tensor_parallel_degree=1 \
	--training.local_batch_size=2 \
	--training.seq_len=1024 \
	--lr_scheduler.warmup_steps=2000 \
	--optimizer.lr=1e-4 \
	--optimizer.eps=1e-8 \
	--model.print_after_conversion \
	--compile.enable \
	--compile.components=model,loss \
	--metrics.enable_wandb \
	--activation_checkpoint.mode=full \
	--debug.moe_force_load_balance \
	--job.dump_folder=/tmp/$@

dbg-intranode-dsv3-16b-mxfp8:
	$(MAKE) intranode-dsv3-16b-mxfp8 DBG=1

intranode-dsv3-16b-mxfp8:
	DBG_ATTACH=$(DBG) $(torchrun_intra) $(ngpu) \
	-m torchtitan.train \
	--job.config_file torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml \
	--metrics.log_freq=1 \
	--training.steps=1500 \
	--parallelism.data_parallel_replicate_degree=1 \
	--parallelism.data_parallel_shard_degree=-1 \
	--parallelism.expert_parallel_degree=2 \
	--parallelism.tensor_parallel_degree=1 \
	--parallelism.expert_tensor_parallel_degree=1 \
	--training.local_batch_size=2 \
	--training.seq_len=1024 \
	--lr_scheduler.warmup_steps=2000 \
	--optimizer.lr=1e-4 \
	--optimizer.eps=1e-8 \
	--model.print_after_conversion \
	--compile.enable \
	--compile.components=model,loss \
	--metrics.enable_wandb \
	--activation_checkpoint.mode=full \
	--debug.moe_force_load_balance \
	--job.dump_folder=/tmp/$@ \
	--model.converters=quantize.grouped_mm.mx,quantize.linear.mx \
	--quantize.grouped_mm.mx.fqns=experts \
	--quantize.grouped_mm.mx.recipe_name=mxfp8 \
	--quantize.linear.mx.recipe_name=mxfp8_cublas_rceil \
	--quantize.linear.mx.filter_fqns=output,router.gate,wq,wkv,wo,feed_forward.w2,shared_experts

# 	--parallelism.expert_parallel_a2a_dispatch_fwd_precision=mxfp8 \
# 	--parallelism.expert_parallel_a2a_combine_bwd_precision=mxfp8 \

h-torchtitan-train:
	python3 -m torchtitan.train --help | less

