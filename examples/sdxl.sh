export MODEL_NAME="checkpoints/stable-diffusion-xl-base-1.0"
export VAE="checkpoints/sdxl-vae-fp16-fix"
export DATASET_NAME="pickapic_v2"
PORT=$((20000 + RANDOM % 10000))

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --main_process_port $PORT --mixed_precision="fp16" --num_processes=8 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=400 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 50 \
  --beta_dpo 5000 \
  --sdxl  \
  --output_dir="ddiminpo-sdxl" 