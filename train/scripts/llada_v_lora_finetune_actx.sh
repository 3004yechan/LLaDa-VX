#!/usr/bin/env bash
# Single-process LoRA finetune for ACT-X (LazySupervisedDataset path, no FIMX/complementary masking).

set -e

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

# Optional checkpoint saving controls (env override)
SAVE_STRATEGY=${SAVE_STRATEGY:-"no"}     # "steps" or "epoch" or "no"
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}     # steps if SAVE_STRATEGY=steps
LOGGING_NAN_INF_FILTER=${LOGGING_NAN_INF_FILTER:-False}

LLM_VERSION="/workspace/model/LLaDA-V-HF"
VISION_MODEL_VERSION="model/siglip2-so400m-patch14-384"

# ACT-X LazySupervisedDataset JSON (converted by tools/convert_actx_to_lazy_supervised.py)
DATA_PATH="/workspace/ACT-X/actX_train_lazy.json"
IMAGE_FOLDER="/workspace/ACT-X"
VIDEO_FOLDER=""

PROMPT_VERSION="llava_llada"
BASE_RUN_NAME="llada_v_lora_actx"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "DATA_PATH: ${DATA_PATH}"
echo "IMAGE_FOLDER: ${IMAGE_FOLDER}"
echo "SAVE_STRATEGY: ${SAVE_STRATEGY}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "LOGGING_NAN_INF_FILTER: ${LOGGING_NAN_INF_FILTER}"

# Use single GPU (0 by default). Set CUDA_VISIBLE_DEVICES before calling if needed.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python llava/train/train_mem.py \
    --model_name_or_path ${LLM_VERSION} \
    --device_map "{\"\":0}" \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_language_model" \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps ${SAVE_INTERVAL} \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_nan_inf_filter ${LOGGING_NAN_INF_FILTER} \
    --tf32 False \
    --model_max_length 4056 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile False \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    --enable_complementary_masking False \
    --use_fimx_dataset False \
    --enable_semi_complementary_masking False

