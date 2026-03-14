#!/bin/bash

# --- Configuration ---
s3_bucket="your-s3-bucket"

 data_name="video_synthetic_static_and_videoFG_v3"
#data_name="video_synthetic_static_v3"

s3_prefix="your-s3-prefix/${data_name}/"
s3_metadata_key="your-s3-prefix/${data_name}/metadata.parquet"

expand_conv_in_init="zero" # or copy
loss_type="pixel"
num_frames=4
mask_cond_mode="vae"
resolution=704
noise_mask_cond=False # ✅ Set this to false to change both the path and the flag
mask_augmentation="all"  # polygon, none, downsample
binary_mask_threshold=127

# --- Conditional Path and Flag Logic ---

# 1. Set a folder name based on the noise_mask_cond flag
if [[ "${noise_mask_cond}" == "True" ]]; then
    noise_mask_folder="with_noise_mask"
else
    noise_mask_folder="without_noise_mask"
fi

# 2. Construct the final output directory using the folder name
output_dir="/path/to/output/stage_2"


# --- Build Command Arguments ---
# Initialize an array with all the static arguments
args=(
    --pretrained_model_name_or_path="/path/to/pretrained_models/stable-video-diffusion-img2vid-xt"
    --per_gpu_batch_size=1
    --gradient_accumulation_steps=8
    --max_train_steps=10000
    --width="${resolution}"
    --height="${resolution}"
    --num_frames "${num_frames}"
    --checkpointing_steps=500
    --checkpoints_total_limit=20
    --learning_rate=5e-5
    --lr_warmup_steps=0
    --seed=42
    --mixed_precision="bf16"
    --max_grad_norm 3.0
    --validation_steps=500
    --s3_bucket "${s3_bucket}"
    --s3_prefix "${s3_prefix}"
    --s3_metadata_key "${s3_metadata_key}"
    --expand_conv_in
    --report_to "wandb"
    --expand_conv_in_init "${expand_conv_in_init}"
    --output_dir "${output_dir}" # Now uses the conditional path
    --loss_type "${loss_type}"
    --mask_cond_mode "${mask_cond_mode}"
    --mask_augmentation "${mask_augmentation}"
    --binary_mask_threshold "${binary_mask_threshold}"
    --l1_weight 1
    --lap_weight 15
    --gradient_weight 0
    --temporal_augmentation_rate 0.5
    --simplification_tolerance 0.005
    --enable_dino_regularization
    --dino_reg_target_module "up_blocks[1].resnets[0]" # mid_block.resnets[0], up_blocks
    --freeze_spatial
    --resume_from_unet_checkpoint "/path/to/output/stage_1/checkpoint-9000"
)
    # --temporal_augmentation_rate
    # --num_occlusions
    # --freeze_spatial
    # --freeze_temporal
    # --resume_from_unet_checkpoint /path/to/checkpoint
    # --trainable_layers "temporal_transformer_block" "attn" "transformer_blocks"

# Conditionally add the --noise_mask_cond flag to the arguments array
if [[ "${noise_mask_cond}" == "True" ]]; then
    args+=(--noise_mask_cond)
fi

# --- Execute the Command ---
# Ensure the output directory exists before launching
mkdir -p "${output_dir}"

echo "🚀 Launching training..."
echo "Output directory: ${output_dir}"

accelerate launch train.py "${args[@]}"