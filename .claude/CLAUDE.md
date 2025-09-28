# Qwen3-Omni SFT Training Project

## Environment Setup
- **Virtual Environment**: `.venv/` (UV-based environment)
- **IMPORTANT**: Always activate the UV environment before running any commands:
  ```bash
  source .venv/bin/activate
  ```

## Main Training Files

### 1. Training Script: `src/train/train_sft.py`
- **Purpose**: Supervised fine-tuning for Qwen3-Omni multimodal model
- **Key Components**:
  - `TextSFTDataset`: Handles JSONL data loading and chat template formatting
  - `ModelArguments`: LoRA configuration (r=64, alpha=16, dropout=0.1)
  - `DataArguments`: Data directory and max sequence length (2048)
  - Uses `Qwen3OmniMoeProcessor` and `Qwen3OmniMoeForConditionalGeneration`
  - Flash Attention 2 implementation with bfloat16 precision

### 2. Training Script: `src/scripts/sft_moe_qwen3omni.sh`
- **Purpose**: Bash script to launch distributed training with DeepSpeed
- **Configuration**:
  - Model: `/home/vladimir_albrekht/projects/2025_sep_22_qwen3omni/ms_swift_training/approach_2_transformers_based/models/5B_small_v`
  - Data: `data/train/*.jsonl`
  - GPUs: 2 (CUDA_VISIBLE_DEVICES=0,1)
  - DeepSpeed Stage 3: `src/train/ds_stage3.json`
  - Batch size: 1 per device, gradient accumulation: 128
  - Learning rate: 1e-5, 3 epochs, cosine scheduler
  - Logging: Wandb integration

## Training Parameters
- **Max sequence length**: 2048
- **Optimizer**: adamw_torch_fused
- **Precision**: bfloat16
- **Gradient checkpointing**: Enabled
- **Save strategy**: Every 500 steps, keep 3 checkpoints

## Data Format
Expected JSONL format with fields:
- `system`: System prompt
- `user`: User message
- `assistant`: Assistant response

## Commands
To run training:
1. Activate environment: `source .venv/bin/activate`
2. Execute: `bash src/scripts/sft_moe_qwen3omni.sh`

## Output
- Timestamped output directories: `output/qwen3omni_sft_run_YYYYMMDD-HHMMSS`
- Model checkpoints and processor saved automatically