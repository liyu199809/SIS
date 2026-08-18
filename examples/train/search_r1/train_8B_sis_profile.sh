#!/bin/bash

# Search-R1 style training with verl-tool
# Copy of train_8B_sis.sh for SIS compute-cost profiling:
#   - enables actor.sis_compute_profiling (coarse, no cuda.synchronize)
#   - exposes actor.topk_k as a configurable hyperparameter
#   - runs only 5 steps
# This script demonstrates how to train an LLM to use search capabilities
PREFIX="${PREFIX:-$HOME/verl_data}"

set -x
model_name="$PREFIX/base_model/Qwen3-8B-Base"
train_data="$PREFIX/data/searchR1_processed_direct/train.parquet"
val_data="$PREFIX/data/searchR1_processed_direct/sub_test.parquet"

rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
use_kl_loss=false
use_kl_in_reward=false
n_gpus_per_node=8
n_nodes=1
n=8
total_epochs=15
total_training_steps=5
batch_size=512
ppo_mini_batch_size=64
max_prompt_length=4096
max_action_length=2048
max_response_length=4096
max_obs_length=1024
temperature=1.0
top_p=1.0
enable_agent=True # enable agent for tool use
strategy="fsdp"
action_stop_tokens="</search>,</answer>"
max_turns=2
kl_loss_coef=0.0002
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=search_r1_qa_em
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=8
tensor_model_parallel_size=1
gpu_memory_utilization=0.6 # higher gpu_memory_utilization will likely cause the vllm to OOM and get stuck, so set it to a lower value like 0.4 or 0.5
do_offload=True # control actor's fsdp.[param|optimizer]_offload and actor_rollout_ref.rollout.fsdp.[param|optimizer]_offload; if gpu_memory_utilization is set to > 0.6, then do_offload should be set to True otherwise it will cause OOM
use_dynamic_bsz=True # faster
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
additional_eos_token_ids=[151645] # <|im_end|> token id
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=False # enable multi-turn training

# ---- SIS profiling knobs ----
# top-K size used by SIS (clip-less) topk / gather and SIS-KL.
topk_k=50
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="grpo-sis-profile-top${topk_k}"
if [ "$enable_agent" = "True" ]; then
    run_name="showD_${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
else
    run_name="showD_${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
fi
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'

# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# Launch the retrieval server and tool server locally (see README for index setup)
file_path=./data/search_r1/retriever_index
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2
python ./verl_tool/servers/tools/utils/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu &
retriever_pid=$!

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
python -m verl_tool.servers.serve --host $host --port $port --tool_type "search_retrieval" --workers_per_tool 8 &
server_pid=$!
tool_server_url=http://$host:$port/get_observation

echo "Server (pid=$server_pid) started at $tool_server_url"
ckpts_home="$PREFIX/IS_ckpts/$reward_manager/$run_name"
token_acceptance_dir="$(pwd)/sis_token_acceptance/$run_name"
mkdir -p "$token_acceptance_dir"
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=2048 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8196 \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
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
    actor_rollout_ref.actor.use_clip_less=True \
    actor_rollout_ref.actor.use_sis_kl=True \
    actor_rollout_ref.actor.topk_k=$topk_k \
    actor_rollout_ref.actor.sis_compute_profiling=True \
    +actor_rollout_ref.actor.log_token_acceptance=True \
    +actor_rollout_ref.actor.token_acceptance_dir=$token_acceptance_dir \
    +actor_rollout_ref.actor.token_acceptance_log_first_epoch_only=True \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.rollout_data_dir=$(pwd)/verl_step_records/$run_name  \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps \


pkill -P -9 $retriever_pid
pkill -P -9 $server_pid
kill -9 $server_pid
