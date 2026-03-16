#!/usr/bin/env bash
# Single-process LoRA finetune for VQA-X on one GPU (no torchrun/DDP).

set -e

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

# FIMX formatting controls
USE_FIMX=${USE_FIMX:-true}
FIMX_ANSWER_BLOCK_SIZE=${FIMX_ANSWER_BLOCK_SIZE:-20}
FIMX_EXPLANATION_FIRST=${FIMX_EXPLANATION_FIRST:-true}
FIMX_EXPLANATION_BLOCK_SIZE=${FIMX_EXPLANATION_BLOCK_SIZE:-50}
ENABLE_SEMI_CM=${ENABLE_SEMI_CM:-true}
ENABLE_CM=${ENABLE_CM:-false}

# Checkpoint / optimizer controls (env override)
SAVE_STRATEGY=${SAVE_STRATEGY:-"no"}   # "steps" or "epoch" or "no"
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}    # steps if SAVE_STRATEGY=steps
LOGGING_NAN_INF_FILTER=${LOGGING_NAN_INF_FILTER:-False}
OPTIM=${OPTIM:-adamw_torch}             # e.g. adamw_torch, paged_adamw_8bit, apollo_adamw
OPTIM_ARGS=${OPTIM_ARGS:-}              # e.g. proj=random,rank=1,scale=128.0,scale_type=tensor,update_proj_gap=200
APOLLO_TARGET_MODULES=${APOLLO_TARGET_MODULES:-"(^model\\.layers\\..*self_attn\\..*)|(^model\\.layers\\..*mlp\\..*)|(^model\\.mm_projector\\..*)|(^lm_head$)|(^base_model\\.model\\.model\\.layers\\..*self_attn\\..*)|(^base_model\\.model\\.model\\.layers\\..*mlp\\..*)|(^base_model\\.model\\.model\\.mm_projector\\..*)|(^base_model\\.model\\.lm_head$)"}
# if train only ,mm_language_model
# APOLLO_TARGET_MODULES="^model\\.layers\\..*self_attn\\..* ^model\\.layers\\..*mlp\\..* ^lm_head$"

LLM_VERSION="/home/user/Yechan/model/LLaDA-V-HF"
VISION_MODEL_VERSION="google/siglip2-so400m-patch14-384"

# User editable paths
DATA_PATH="/home/user/Yechan/Dataset/VQA-X/vqaX_train_fimx.json"
IMAGE_FOLDER="/home/user/Yechan/Dataset/VQA-X"
VIDEO_FOLDER=""

PROMPT_VERSION="llava_llada"
BASE_RUN_NAME="llada_v_lora_vqax_single"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "DATA_PATH: ${DATA_PATH}"
echo "IMAGE_FOLDER: ${IMAGE_FOLDER}"
echo "USE_FIMX: ${USE_FIMX}"
echo "FIMX_ANSWER_BLOCK_SIZE: ${FIMX_ANSWER_BLOCK_SIZE}"
echo "FIMX_EXPLANATION_FIRST: ${FIMX_EXPLANATION_FIRST}"
echo "FIMX_EXPLANATION_BLOCK_SIZE: ${FIMX_EXPLANATION_BLOCK_SIZE}"
echo "ENABLE_SEMI_CM: ${ENABLE_SEMI_CM}"
echo "ENABLE_CM: ${ENABLE_CM}"
echo "SAVE_STRATEGY: ${SAVE_STRATEGY}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "LOGGING_NAN_INF_FILTER: ${LOGGING_NAN_INF_FILTER}"
echo "OPTIM: ${OPTIM}"
echo "OPTIM_ARGS: ${OPTIM_ARGS}"

EXTRA_OPTIM_ARGS=()
if [[ "${OPTIM}" == apollo_* || "${OPTIM}" == galore_* ]]; then
    EXTRA_OPTIM_ARGS+=(--optim_target_modules "${APOLLO_TARGET_MODULES}")
    if [[ -n "${OPTIM_ARGS}" ]]; then
        EXTRA_OPTIM_ARGS+=(--optim_args "${OPTIM_ARGS}")
    fi
fi

# Use single GPU (0 by default). Set CUDA_VISIBLE_DEVICES before calling if needed.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python llava/train/train_mem.py \
    --model_name_or_path ${LLM_VERSION} \
    --device_map '{"":0}' \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --run_name $BASE_RUN_NAME \
    --output_dir "exp/$BASE_RUN_NAME" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps ${SAVE_INTERVAL} \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_nan_inf_filter ${LOGGING_NAN_INF_FILTER} \
    --optim ${OPTIM} \
    "${EXTRA_OPTIM_ARGS[@]}" \
    --tf32 False \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile False \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    --enable_complementary_masking ${ENABLE_CM} \
    --use_fimx_dataset ${USE_FIMX} \
    --fimx_answer_block_size ${FIMX_ANSWER_BLOCK_SIZE} \
    --fimx_explanation_first ${FIMX_EXPLANATION_FIRST} \
    --fimx_explanation_block_size ${FIMX_EXPLANATION_BLOCK_SIZE} \
    --enable_semi_complementary_masking ${ENABLE_SEMI_CM}
