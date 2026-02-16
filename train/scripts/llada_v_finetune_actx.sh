#!/usr/bin/env bash
# Full finetune script for ACT-X (based on original llada_v_finetune.sh).

set -e

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

num_node=$1
gpu_num=$1

# Run examples:
#   Single node, single GPU: bash scripts/llada_v_finetune_actx.sh 1 1
#   Single node, 8 GPUs:      bash scripts/llada_v_finetune_actx.sh 1 8

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

# Checkpoint saving controls
SAVE_STRATEGY=${SAVE_STRATEGY:-"no"}   # "steps" or "epoch" or "no"
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}      # steps if SAVE_STRATEGY=steps, ignored otherwise
LOGGING_NAN_INF_FILTER=${LOGGING_NAN_INF_FILTER:-False}
echo "save_strategy ${SAVE_STRATEGY}"
echo "save_interval ${SAVE_INTERVAL}"
echo "logging_nan_inf_filter ${LOGGING_NAN_INF_FILTER}"

LLM_VERSION="/workspace/model/LLaDA-V-HF"
VISION_MODEL_VERSION="model/siglip2-so400m-patch14-384"

############### Finetune (ACT-X LazySupervisedDataset) ################

PROMPT_VERSION="llava_llada"
BASE_RUN_NAME="llada_v_finetune_actx"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

DATA_PATH="/workspace/ACT-X/actX_train_lazy.json"
IMAGE_FOLDER="/workspace/ACT-X"
VIDEO_FOLDER=""
echo "data_path ${DATA_PATH}"
echo "image_folder ${IMAGE_FOLDER}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "exp/$BASE_RUN_NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps ${SAVE_INTERVAL} \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_nan_inf_filter ${LOGGING_NAN_INF_FILTER} \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    --use_fimx_dataset False \
    --enable_complementary_masking False \
    --enable_semi_complementary_masking False

