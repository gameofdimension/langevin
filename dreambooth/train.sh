#!/usr/bin/env bash

# source
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb

# requirements
# !pip install xformers bitsandbytes transformers accelerate diffusers wandb -q

# prepare
# accelerate config default
# huggingface-cli login
# wandb login

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_data_dir="dog" \
  --output_dir="lora-trained-xl-colab" \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --use_8bit_adam \
  --max_train_steps=1000 \
  --checkpointing_steps=100 \
  --seed="0" \
  --report_to="wandb" \
  --resume_from_checkpoint="latest" \
  --validation_prompt="a photo of sks dog on the grass" \
  --num_validation_images=1 \
  --push_to_hub
