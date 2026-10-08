#!/bin/bash

set -euo pipefail

export PYTHONWARNINGS="ignore"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Distributed runtime
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29993}"
export WANDB_PROJECT="${WANDB_PROJECT:-vlm2-vsibench}"

export NCCL_TIMEOUT=36000

# VSI-Bench data
# Dataset paths are machine-specific and must be supplied by the user.
: "${IMAGE_FOLDER:?Set IMAGE_FOLDER to the processed VSI-Bench data directory}"
VIDEO_FOLDER="${VIDEO_FOLDER:-${IMAGE_FOLDER}}"
DATA_YAML="${DATA_YAML:-${SCRIPT_DIR}/all_h_vsibench_data.yaml}"

# Experiment
SUFFIX="${RUN_NAME:-vlm2_vsibench}"
NUM_TRAIN_EPOCHS=5
SAVE_TOTAL_LIMIT=5

# Model configuration
FUSION_BLOCK="cross_attention"
SPATIAL_TOWER="pi3_vaga"
SPATIAL_TOWER_SELECT_FEATURE="patch_tokens"
SPATIAL_FEATURE_DIM="2048"
TUNE_MM_MLP_ADAPTER=True
GRADIENT_ACCUMULATION_STEPS=16 # bs=128, num_gpus=8
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"

# Checkpoints and outputs
MID_RUN_NAME="llava_video_7b_qwen2_${SUFFIX}"
PREV_STAGE_CHECKPOINT="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/work_dirs_ours}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs_ours}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# Training
ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nproc_per_node="${NUM_GPUS_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2_memory.json \
    --model_name_or_path "${PREV_STAGE_CHECKPOINT}" \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --spatial_tower "${SPATIAL_TOWER}" \
    --spatial_tower_select_feature "${SPATIAL_TOWER_SELECT_FEATURE}" \
    --spatial_feature_dim "${SPATIAL_FEATURE_DIM}" \
    --fusion_block "${FUSION_BLOCK}" \
    --tune_spatial_tower False \
    --tune_fusion_block True \
    --tune_mm_mlp_adapter "${TUNE_MM_MLP_ADAPTER}" \
    --version "${PROMPT_VERSION}" \
    --data_path "${DATA_YAML}" \
    --image_folder "${IMAGE_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name "${SUFFIX}" \
    --output_dir "${OUTPUT_ROOT}/${MID_RUN_NAME}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --verbose_logging True \
    --spatial_feature_dirname None \
    --save_spatial_features False \
    --use_memory_block True \
    --world_position_embedding_type norm-avg-discrete-mlp-mask-memory \
    2>&1 | tee "${LOG_DIR}/${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
