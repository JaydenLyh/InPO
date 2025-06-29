export MODEL_NAME="checkpoints/stable-diffusion-v1-5"
export DATASET_NAME="datasets/pickapic_v2"
PORT=$((20000 + RANDOM % 10000))

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --main_process_port $PORT --mixed_precision="fp16" --num_processes=8 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --unet_init="checkpoints/sd15-sft" \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=400 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 500 \
  --beta_dpo 2000 \
  --output_dir="ddiminpo-sd15" \