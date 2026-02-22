# LLaDa-VX

This repository contains our LLaDa-VX training and demo code.
This repository is forked from https://github.com/ML-GSAI/LLaDA-V.

## Environment Setup

Use the following commands to prepare the environment:

```bash
python -m venv ~/venvs/llada-vx
source ~/venvs/llada-vx/bin/activate
# pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
cd LLaDA-VX/train
bash init_env.sh
```

## Setup Check

If the following command runs successfully, the environment setup is complete:

```bash
python generate_demo.py
```

## Training

We trained the model with:

```bash
scripts/llada_v_lora_finetune_actx.sh
```

## Dataset and Evaluation

For dataset preparation and evaluation details, please refer to:

- https://github.com/fawazsammani/nlxgpt
