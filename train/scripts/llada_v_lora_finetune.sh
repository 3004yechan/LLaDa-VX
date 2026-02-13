#!/usr/bin/env bash
# Distributed LoRA finetune (torchrun + DeepSpeed, no 4bit quantization).

set -e

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

num_node=$1
gpu_num=$2
# Optional third arg to toggle complementary masking (default: false)
enable_cm=${3:-false}
# Optional fourth arg to toggle FIMX dataset loader (default: false)
enable_fimx=${4:-false}
# Optional fifth arg for FIMX answer_block_size (default: 20)
fimx_answer_block_size=${5:-20}
# Optional sixth arg to toggle semi-complementary masking (default: false)
enable_semi_cm=${6:-false}

# Optional checkpoint saving controls (env override)
SAVE_STRATEGY=${SAVE_STRATEGY:-"steps"}   # "steps" or "epoch" or "no"
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}      # steps if SAVE_STRATEGY=steps, ignored otherwise

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"
echo "enable_complementary_masking ${enable_cm}"
echo "use_fimx_dataset ${enable_fimx}"
echo "fimx_answer_block_size ${fimx_answer_block_size}"
echo "enable_semi_complementary_masking ${enable_semi_cm}"
echo "save_strategy ${SAVE_STRATEGY}"
echo "save_interval ${SAVE_INTERVAL}"

LLM_VERSION="/home/20223206/model/LLaDA-V-HF"
VISION_MODEL_VERSION="model/siglip2-so400m-patch14-384"

# User editable paths
DATA_PATH="/home/20223206/ACT-X/actX_train_filled_llada_trim.json"
IMAGE_FOLDER="/home/20223206/ACT-X"
VIDEO_FOLDER=""

PROMPT_VERSION="llava_llada"
BASE_RUN_NAME="llada_v_lora"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
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
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps ${SAVE_INTERVAL} \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile False \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    --enable_complementary_masking ${enable_cm} \
    --use_fimx_dataset ${enable_fimx} \
    --fimx_answer_block_size ${fimx_answer_block_size} \
    --enable_semi_complementary_masking ${enable_semi_cm}
