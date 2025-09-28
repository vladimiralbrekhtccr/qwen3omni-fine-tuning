#!/bin/bash
set -e
set -u

export TRITON_CACHE_DIR=".cache/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

if [ -f .env ]; then
  set -a # Automatically export all variables
  source .env
  set +a # Stop automatically exporting
fi

MODEL_PATH="/home/vladimir_albrekht/projects/2025_sep_22_qwen3omni/ms_swift_training/approach_2_transformers_based/models/5B_small_v"
DATA_DIR="data/train"
OUTPUT_DIR="output/qwen3omni_sft_run_$(date +%Y%m%d-%H%M%S)"
DEEPSPEED_CONFIG="src/train/ds_stage3.json"

mkdir -p "$OUTPUT_DIR"
WANDB_RUN_NAME=$(basename "$OUTPUT_DIR")
NUM_GPUS=4

echo "--- Starting SFT Run ---"
echo "Model Path: ${MODEL_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "--------------------------"

# For debugging with 3 samples, use minimal settings
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=${NUM_GPUS} src/train/train_sft.py \
  --model_path_or_name "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${WANDB_RUN_NAME}" \
  --max_seq_length 2048 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config '{"use_orig_params": true}' \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --max_steps 30 \
  --lr_scheduler_type "constant" \
  --warmup_steps 0 \
  --weight_decay 0.01 \
  --bf16 True \
  --gradient_checkpointing True \
  --save_strategy "no" \
  --logging_steps 1 \
  --report_to "wandb" \
  --optim "adamw_torch" \
  --dataloader_num_workers 0

echo "--- SFT run completed. Checkpoints are saved in ${OUTPUT_DIR} ---"