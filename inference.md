### Directory Structure

The script expects a specific directory structure for your input data. You need a root folder for images and another for masks. Inside each, there should be subdirectories for each video, with matching names.

```
/path/to/your/dataset/
├── image/
│   ├── video_001/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── video_002/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│
└── mask/
    ├── video_001/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    ├── video_002/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
```

* The names of the image files and mask files within the corresponding video folders must match (e.g., `0000.png` in images corresponds to `0000.png` in masks).

### Usage

Run the script from your terminal. Below is the general usage format and a detailed breakdown of all available arguments.

#### Basic Command

```bash
python inference_onestep_folder.py \
    --base_model_path "/path/to/your/stable-video-diffusion-img2vid-xt" \
    --unet_checkpoint_path "/path/to/your/unet_checkpoint" \
    --image_root_path "/path/to/image/folder" \
    --mask_root_path "/path/to/mask/folder" \
    --output_dir "/path/to/save/results" \
    [--optional_arguments]
```

#### Command-Line Arguments

##### **Paths**
* `--base_model_path`: Path to the base SVD model directory. (Default: `/path/to/pretrained_models/stable-video-diffusion-img2vid-xt`)
* `--unet_checkpoint_path`: **(Required)** Path to the fine-tuned UNet checkpoint.
* `--image_root_path`: **(Required)** Root folder containing input image sequences.
* `--mask_root_path`: **(Required)** Root folder containing input mask sequences.
* `--output_dir`: Directory to save all outputs. (Default: `output_batch`)

#### **Inference Configuration**
* `--num_frames`: Number of frames to generate. (Default: 16)
* `--num_input_frames`: Number of frames to read from input folders. (Default: same as `num_frames`)
* `--width`: Processing width for the frames. (Default: 1024)
* `--height`: Processing height for the frames. (Default: 576)
* `--keep_aspect_ratio`: If set, maintains the aspect ratio of the input images.
* `--mask_cond_mode`: Mask conditioning mode. (Choices: `vae`, `interpolate`, Default: `vae`)
* `--mixed_precision`: Use mixed precision for inference. (Choices: `no`, `fp16`, `bf16`, Default: `fp16`)
* `--seed`: A seed for reproducibility. (Default: 42)

#### **Mask Augmentation**
* `--mask_augmentation`: Type of augmentation to apply to the masks. (Choices: `none`, `polygon`, `downsample`, `bounding_box`, Default: `none`)
* `--downsample_factor`: Downsampling factor if `mask_augmentation` is `downsample`. (Default: 8)
* `--simplification_tolerance`: Simplification tolerance for `polygon` augmentation. (Default: 0.001)
* `--save_processed_mask`: If set, saves the final augmented masks that were fed into the model.

#### **Temporal Augmentation**
* `--temporal_augmentation`: If set, applies diverse temporal augmentations (occlusions, erosions, etc.) to random mask frames.
* `--num_occlusions`: Number of frames to apply temporal augmentation to. (Default: 1)
* `--occlusion_shape`: Shape for the temporal occlusion operation. (Choices: `rectangle`, `circle`, Default: `rectangle`)
* `--occlusion_scale_range`: Scale range for the occlusion, relative to the mask's bounding box. (Default: `[0.2, 0.5]`)
* `--erosion_dilation_kernel_size`: Kernel size for erosion and dilation operations. (Default: 5)

---

## Output Structure

The script will generate an output directory with the following structure:

```
/path/to/your/output_dir/
├── mask_guide/ (if --save_processed_mask is used)
│   ├── video_001/
│   │   ├── frame_0000.png
│   │   └── ...
│   └── video_002/
│       └── ...
│
└── results/
    ├── video_001/
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   ├── ...
    │   └── video.mp4 (if more than one frame is generated)
    └── video_002/
        └── ...
```