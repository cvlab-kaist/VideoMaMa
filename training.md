### Summary of Key Training Parameters (Two-Stage Training)

This document outlines the two-stage training process. **Stage 1** focuses on training the spatial components of the model on single frames. **Stage 2** builds on this by freezing the spatial layers and fine-tuning the temporal components on video clips.

We utilized AWS S3 for data storage during training. However, if you have sufficient local storage, you can configure the code to run locally without S3.

---

### **Stage 1: Spatial Layer Training (`train_stage1.sh`)**

This initial stage trains the model's spatial understanding using single frames and freezes the temporal layers.

#### **Configuration in `train_stage1.sh`**
-   **Dataset:**
    -   `s3_bucket`: "your-s3-bucket"
    -   `data_name`: "video_synthetic_static"
-   **Model & Training Hyperparameters:**
    -   `num_frames`: 1
    -   `resolution`: 1024
    -   `loss_type`: "pixel"
    -   `mask_cond_mode`: "vae"
    -   `temporal_augmentation_rate`: 0.0
-   **Output Directory:**
    -   `output_dir`: "/path/to/output/stage_1"

#### **Key Python Arguments & Their Values (Stage 1)**

-   `--freeze_temporal`
    -   **Purpose:** Freezes all temporal layers in the UNet. This ensures that only the spatial components are trained, adapting the model to the target domain on a per-frame basis.
-   `--num_frames 1`
    -   **Purpose:** Specifies that each training sample consists of only a single frame, which is appropriate for training spatial features without temporal context.
-   `--enable_dino_regularization`
    -   **Purpose:** Activates the DINO feature alignment loss to regularize the spatial features being trained.
-   `--dino_reg_target_module "up_blocks[1].resnets[0]"`
    -   **Purpose:** Specifies the exact UNet layer whose features will be aligned with DINO's features.
-   `--temporal_augmentation_rate 0.0`
    -   **Purpose:** Disables temporal augmentations, as they are not relevant when training on single frames.

---

### **Stage 2: Temporal Layer Training (`train_stage2.sh`)**

This second stage resumes from the Stage 1 checkpoint, freezes the now-trained spatial layers, and focuses exclusively on training the temporal layers using video clips.

#### **Configuration in `train_stage2.sh`**

-   **Dataset:**
    -   `s3_bucket`: "your-s3-bucket"
    -   `data_name`: "video_synthetic_static_and_videoFG_v3"
-   **Model & Training Hyperparameters:**
    -   `num_frames`: 4
    -   `resolution`: 704
    -   `loss_type`: "pixel"
    -   `mask_cond_mode`: "vae"
    -   `temporal_augmentation_rate`: 0.5
-   **Output Directory:**
    -   `output_dir`: "/path/to/output/stage_2"

#### **Key Python Arguments & Their Values (Stage 2)**

-   `--resume_from_unet_checkpoint "/path/to/output/stage_1/checkpoint"`
    -   **Purpose:** Loads the UNet weights from the completed Stage 1 training run, providing a strong starting point for the spatial features.
-   `--freeze_spatial`
    -   **Purpose:** Freezes all non-temporal (spatial) layers in the UNet. This is the inverse of Stage 1 and ensures that only the temporal components learn from the video clips.
-   `--num_frames 4`
    -   **Purpose:** Sets the number of frames per video clip to 4, allowing the model to learn temporal relationships.
-   `--enable_dino_regularization` & `--dino_reg_target_module "up_blocks[1].resnets[0]"`
    -   **Purpose:** Continues to use DINO regularization, which now helps guide the temporal layers to produce feature maps that are consistent over time.
-   `--temporal_augmentation_rate 0.5`
    -   **Purpose:** Enables temporal augmentations with a 50% probability, helping the model generalize better to different video dynamics.

---

### Important Python Arguments & Their Values

Below are the key arguments passed to `train.py` and their explanations.

#### **Training Strategy Arguments**

These arguments define this as a Stage 2 training run, resuming from a previous checkpoint and freezing spatial layers.

-   `--resume_from_unet_checkpoint "/path/to/output/stage_1/checkpoint"`
    -   **Purpose:** Resumes training by loading the UNet weights from a specified Stage 1 checkpoint.
-   `--freeze_spatial`
    -   **Purpose:** Freezes all non-temporal (spatial) layers in the UNet. This ensures that only the temporal components are trained in this stage.

#### **DINO Regularization Arguments**

These flags enable and configure the DINO feature alignment loss.

-   `--enable_dino_regularization`
    -   **Purpose:** Activates the DINO feature alignment loss, which encourages the UNet's features to match the DINO model's features.
-   `--dino_reg_target_module "up_blocks[1].resnets[0]"`
    -   **Purpose:** Specifies the exact layer within the UNet (`up_blocks[1].resnets[0]`) from which to extract intermediate features for the alignment loss.

#### **Model and Loss Arguments**

These define the model architecture, loss function, and conditioning.

-   `--loss_type "pixel"` or `--loss_type "mse"` 
    -   **Purpose:** Sets the training loss to be the pixel-space matting loss, which is a combination of L1 and Laplacian losses.
-   `--mask_cond_mode "vae"`
    -   **Purpose:** Specifies that the conditioning mask will be processed by encoding it with the VAE into the latent space.
-   `--l1_weight 1` & `--lap_weight 15`
    -   **Purpose:** Sets the weights for the L1 and Laplacian components of the pixel-space matting loss.

#### **Data and Augmentation Arguments**

-   `--dataset_type s3`
    -   **Purpose:** Choose whether we will train with the dataset that have already generate or make it on the fly.
-   `--num_frames 4`
    -   **Purpose:** Sets the number of frames per video clip to be used in training.
-   `--resolution`
    -   **Purpose:** Sets the resolution of frames to be used in training.
-   `--mask_augmentation "all"`
    -   **Purpose:** Applies all available mask augmentations (e.g., polygon simplification, downsampling) during training.
-   `--temporal_augmentation_rate 0.5`
    -   **Purpose:** Defines the probability (50%) of applying temporal augmentations to each training sample.