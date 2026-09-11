#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/train.sh CONFIG.env {deco|pdn} [Hydra overrides ...]" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
CONFIG_FILE=$1
REWARD_METHOD=$2
shift 2

[[ -f "$CONFIG_FILE" ]] || { echo "Configuration file not found: $CONFIG_FILE" >&2; exit 1; }
[[ "$REWARD_METHOD" == "deco" || "$REWARD_METHOD" == "pdn" ]] || usage

set -a
source "$CONFIG_FILE"
set +a

required=(VERL_DIR MODEL_PATH TRAIN_FILE VAL_FILE OUTPUT_DIR CACHE_DIR GPU_LIST TRAIN_JUDGE_MODEL)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing required setting: $name" >&2; exit 1; }
  [[ "${!name}" != *"/path/to/"* && "${!name}" != "provider/model-name" ]] || {
    echo "Replace the placeholder value for $name" >&2
    exit 1
  }
done

[[ -d "$VERL_DIR" ]] || { echo "VERL_DIR is not a directory: $VERL_DIR" >&2; exit 1; }
[[ -f "$TRAIN_FILE" ]] || { echo "TRAIN_FILE not found: $TRAIN_FILE" >&2; exit 1; }
[[ -f "$VAL_FILE" ]] || { echo "VAL_FILE not found: $VAL_FILE" >&2; exit 1; }
[[ -f "$VERL_DIR/recipe/humanlm/reward_function.py" ]] || {
  echo "HumanLM is not installed at $VERL_DIR/recipe/humanlm" >&2
  exit 1
}

VERL_DIR=$(cd "$VERL_DIR" && pwd)
TRAIN_FILE=$(realpath "$TRAIN_FILE")
VAL_FILE=$(realpath "$VAL_FILE")
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
CACHE_DIR=$(realpath "$CACHE_DIR")

PYTHON_BIN=${PYTHON_BIN:-python3}
EVAL_JUDGE_MODEL=${EVAL_JUDGE_MODEL:-$TRAIN_JUDGE_MODEL}
ROLLOUTS=${ROLLOUTS:-4}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-5120}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-0.8}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
KL_COEFFICIENT=${KL_COEFFICIENT:-0.001}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-50}
SEED=${SEED:-0}
LOGGER_JSON=${LOGGER_JSON:-'["console"]'}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"deco_${REWARD_METHOD}"}
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_LIST")

if [[ "$REWARD_METHOD" == "pdn" && "$ROLLOUTS" -lt 2 ]]; then
  echo "PDN requires ROLLOUTS >= 2" >&2
  exit 1
fi

STATE_CONFIG=./recipe/humanlm/state_config/sebvgcr.json
DIMENSION_CONFIG=./recipe/humanlm/state_config/sebvgc.json
CHAT_TEMPLATE=./recipe/humanlm/chat_templates/qwen3_multi_role_template_think.jinja
TRAIN_METRICS="{response:{state_reward_on_response:{weight:1.0,kwargs:{model:\"$TRAIN_JUDGE_MODEL\",temperature:0,config_path:\"$DIMENSION_CONFIG\"}}}}"
VAL_METRICS="{response:{state_reward_on_response:{weight:1.0,kwargs:{model:\"$EVAL_JUDGE_MODEL\",temperature:0,config_path:\"$DIMENSION_CONFIG\"}}}}"
STATE_CONFIG_ABS="$VERL_DIR/recipe/humanlm/state_config/sebvgcr.json"
STOP_SEQUENCES=$("$PYTHON_BIN" -c "import json; print(json.dumps([f'</{k}>' for k in json.load(open('$STATE_CONFIG_ABS'))]))")

export CUDA_VISIBLE_DEVICES=$GPU_LIST
export VLLM_USE_V1=${VLLM_USE_V1:-1}

echo "Model: $MODEL_PATH"
echo "Reward: $REWARD_METHOD"
echo "GPUs: $GPU_LIST"
echo "Training data: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"

cd "$VERL_DIR"
command=(
  "$PYTHON_BIN" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  trainer.val_before_train=False
  +trainer.load_state_map=False
  reward.reward_manager.source=importlib
  reward.reward_manager.name=HumanLMRewardManager
  reward.reward_manager.module.path=./recipe/humanlm/reward_function.py
  custom_reward_function.path=./recipe/humanlm/reward_function.py
  custom_reward_function.name=compute_reward
  +reward_model.reward_kwargs.enable_state=False
  +reward_model.reward_kwargs.fetch_global_best_state=True
  +reward_model.reward_kwargs.separate_generation=True
  +reward_model.reward_kwargs.enable_thinking=True
  +reward_model.reward_kwargs.state_config="$STATE_CONFIG"
  +reward_model.reward_kwargs.strict_format=True
  +reward_model.reward_kwargs.train_metrics="$TRAIN_METRICS"
  +reward_model.reward_kwargs.val_metrics="$VAL_METRICS"
  +reward_model.reward_kwargs.response_reward_aggregation="$REWARD_METHOD"
  +reward_model.reward_kwargs.pdn_epsilon=1e-6
  +reward_model.reward_kwargs.n_rollouts="$ROLLOUTS"
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  +data.cache_dir="$CACHE_DIR"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.val_batch_size="$TRAIN_BATCH_SIZE"
  +data.kwargs.multirole_chat_template_path="$CHAT_TEMPLATE"
  +data.seed="$SEED"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.filter_overlong_prompts=False
  data.truncation=error
  +data.state_config_path="$STATE_CONFIG"
  +data.enable_hetero_think=True
  +data.augment_with_states=True
  +data.dataset=reddit
  +data.custom_cls.path=./recipe/humanlm/state_dataset.py
  +data.custom_cls.name=StateDataset
  +data.additional_generation_prompt=''
  +data.apply_chat_template_kwargs.enable_thinking=True
  +actor_rollout_ref.model.custom_chat_template="$CHAT_TEMPLATE"
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.lora_rank="$LORA_RANK"
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
  actor_rollout_ref.model.target_modules="$LORA_TARGET_MODULES"
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEFFICIENT"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n="$ROLLOUTS"
  actor_rollout_ref.rollout.temperature="$TRAIN_TEMPERATURE"
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  +actor_rollout_ref.rollout.engine_kwargs.dtype=bfloat16
  "+actor_rollout_ref.rollout.engine_kwargs={stop: $STOP_SEQUENCES}"
  actor_rollout_ref.rollout.agent.default_agent_loop=humanlm_agent
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.humanlm.humanlm_agent_loop_worker.HumanLMAgentLoopManager
  +actor_rollout_ref.rollout.agent.agent_loop_config_path=./recipe/humanlm/configs/humanlm_agent_loop_config.yaml
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  trainer.resume_mode=auto
  trainer.n_gpus_per_node="$NUM_GPUS"
  trainer.nnodes=1
  trainer.logger="$LOGGER_JSON"
  trainer.project_name=deco-humanlm
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.default_local_dir="$OUTPUT_DIR"
  trainer.save_freq="$SAVE_FREQUENCY"
  trainer.test_freq="$SAVE_FREQUENCY"
  trainer.default_hdfs_dir=null
  trainer.total_epochs="$TOTAL_EPOCHS"
)

if [[ "$TOTAL_TRAINING_STEPS" != "null" ]]; then
  command+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi
if [[ "$LORA_RANK" -gt 0 ]]; then
  command+=("actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_adapter]")
fi
command+=("$@")
"${command[@]}"
