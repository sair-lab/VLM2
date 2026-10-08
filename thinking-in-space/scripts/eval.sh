#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname -- "$EVAL_ROOT")"
MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: bash $0 /absolute/path/to/checkpoint" >&2
    exit 2
fi
cd "$EVAL_ROOT"
export PYTHONPATH="$EVAL_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LMMS_EVAL_LAUNCHER=accelerate

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29676}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
MODEL_NAME="${MODEL_NAME:-llava_video_7b_qwen2_vlm2_lora}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
RUN_NAME="${RUN_NAME:-vlm2_vsibench}"
OUTPUT_PATH="${OUTPUT_PATH:-${EVAL_ROOT}/logs/$(date +%Y%m%d)/vsibench}"

python -m accelerate.commands.launch \
    --num_processes="$NUM_GPUS_PER_NODE" \
    --main_process_port="$MASTER_PORT" \
    -m lmms_eval \
    --model vlm2 \
    --model_args "pretrained=$MODEL_PATH,model_base=$MODEL_BASE,model_name=$MODEL_NAME,conv_template=qwen_1_5,max_frames_num=$MAX_FRAMES_NUM,video_decode_backend=decord" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "$RUN_NAME" \
    --output_path "$OUTPUT_PATH"
