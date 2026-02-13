#!/usr/bin/env bash
# Single-process QLoRA finetune on one GPU (no torchrun/DDP).

set -e

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

# Optional toggles (set via environment variables before running)
ENABLE_CM=${ENABLE_CM:-false}            # complementary masking (mask + inverse mask)
USE_FIMX=${USE_FIMX:-false}              # use LazySupervisedDatasetForFIMX
FIMX_ANSWER_BLOCK_SIZE=${FIMX_ANSWER_BLOCK_SIZE:-20}  # answer block size for FIMX
ENABLE_SEMI_CM=${ENABLE_SEMI_CM:-false}  # semi-complementary masking for FIMX (answer always masked, prefix/because never masked)
# Checkpoint saving controls (env override)
SAVE_STRATEGY=${SAVE_STRATEGY:-"no"}   # "steps" or "epoch" or "no"
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}      # steps if SAVE_STRATEGY=steps, ignored otherwise

# Paths to edit
LLM_VERSION="/home/20223206/model/LLaDA-V-HF"  # local HF-style model dir
VISION_MODEL_VERSION="model/siglip2-so400m-patch14-384"
DATA_PATH="/home/20223206/ACT-X/actX_train_filled_llada_trim.json"
IMAGE_FOLDER="/home/20223206/ACT-X"
VIDEO_FOLDER=""

PROMPT_VERSION="llava_llada"
BASE_RUN_NAME="llada_v_qlora_single"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "ENABLE_CM: ${ENABLE_CM}"
echo "USE_FIMX: ${USE_FIMX}"
echo "FIMX_ANSWER_BLOCK_SIZE: ${FIMX_ANSWER_BLOCK_SIZE}"
echo "ENABLE_SEMI_CM: ${ENABLE_SEMI_CM}"
echo "SAVE_STRATEGY: ${SAVE_STRATEGY}"
echo "SAVE_INTERVAL: ${SAVE_INTERVAL}"

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
    --bits 4 \
    --quant_type nf4 \
    --double_quant True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --run_name $BASE_RUN_NAME \
    --output_dir "exp/$BASE_RUN_NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps ${SAVE_INTERVAL} \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile False \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    --enable_complementary_masking ${ENABLE_CM} \
    --use_fimx_dataset ${USE_FIMX} \
    --fimx_answer_block_size ${FIMX_ANSWER_BLOCK_SIZE} \
    --enable_semi_complementary_masking ${ENABLE_SEMI_CM}
