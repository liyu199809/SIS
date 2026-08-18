#!/usr/bin/env bash
# GRPO training for math reasoning (Qwen3-8B-Base, no tool use).
#   - Train data: DAPO-Math-17k (preprocessed via examples/data_preprocess/math_simple_rl.py)
#   - Val data:   MATH-500 (Avg@1) + AMC23 / AIME24 / AIME25 (Avg@32)
#   - Reward:     verl_tool reward_manager `math_dapo` (strict_box_verify=True)
#   - Val n is per-row, controlled by extra_info["n_samples"] (see _validate hook).
set -x

#######################################
# paths
#######################################
DATA_ROOT="${DATA_ROOT:-$HOME/verl_data/data/math_simple_rl}"
train_data=${DATA_ROOT}/train.parquet
val_data="[${DATA_ROOT}/math500_test.parquet,${DATA_ROOT}/amc23_test.parquet,${DATA_ROOT}/aime24_test.parquet,${DATA_ROOT}/aime25_test.parquet]"

model_name="${MODEL_NAME:-$HOME/verl_data/base_model/Qwen3-8B-Base}"
ckpt_root="${CKPT_ROOT:-$HOME/verl_data/ckpts}"

#######################################
# core hp
#######################################
rl_alg=grpo
loss_mode=cispo
use_clip_less=True
n_gpus_per_node=8
n_nodes=1
total_training_steps=400

# rollout group
n=8                     # GRPO group size (training rollouts per prompt)
batch_size=512
ppo_mini_batch_size=256
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=4

max_prompt_length=1024
max_response_length=3072

# training rollout sampling
temperature=1.0
top_p=1.0

# val rollout sampling (Avg@k semantics)
val_temperature=1.0
val_top_p=0.95
val_n_default=1         # fallback when extra_info.n_samples missing; per-row n_samples controls real k

lr=1e-6
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl

reward_manager=math_dapo
strategy=fsdp
tensor_model_parallel_size=2
gpu_memory_utilization=0.7
do_offload=True
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1

#######################################
# run id
#######################################
model_pretty_name=$(echo $model_name | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')
if [ "$use_clip_less" = "True" ]; then
    suffix="${loss_mode}-sis"
else
    suffix="${loss_mode}"
fi
run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-${suffix}-newvals"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1

mkdir -p ${ckpt_root}/${run_name} ./logs

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.use_clip_less=$use_clip_less \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=$val_n_default \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=topo_math \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_local_dir=${ckpt_root}/${run_name} \
    trainer.default_hdfs_dir=null \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.total_training_steps=$total_training_steps \


