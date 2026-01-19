"""
VideoMaMa Inference Wrapper - Hugging Face Space Version
Handles video matting with mask conditioning
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from typing import List

from pipeline_svd_mask import VideoInferencePipeline


def videomama(pipeline, frames_np, mask_frames_np):
    """
    Run VideoMaMa inference on video frames with mask conditioning
    
    Args:
        pipeline: VideoInferencePipeline instance
        frames_np: List of numpy arrays, [(H,W,3)]*n, uint8 RGB frames
        mask_frames_np: List of numpy arrays, [(H,W)]*n, uint8 grayscale masks
        
    Returns:
        output_frames: List of numpy arrays, [(H,W,3)]*n, uint8 RGB outputs
    """
    # Convert numpy arrays to PIL Images
    frames_pil = [Image.fromarray(f) for f in frames_np]
    mask_frames_pil = [Image.fromarray(m, mode='L') for m in mask_frames_np]
    
    # Resize to model input size
    target_width, target_height = 1024, 576
    frames_resized = [f.resize((target_width, target_height), Image.Resampling.BILINEAR) 
                     for f in frames_pil]
    masks_resized = [m.resize((target_width, target_height), Image.Resampling.BILINEAR) 
                    for m in mask_frames_pil]
    
    # Run inference
    print(f"Running VideoMaMa inference on {len(frames_resized)} frames...")
    output_frames_pil = pipeline.run(
        cond_frames=frames_resized,
        mask_frames=masks_resized,
        seed=42,
        mask_cond_mode="vae"
    )
    
    # Resize back to original resolution
    original_size = frames_pil[0].size
    output_frames_resized = [f.resize(original_size, Image.Resampling.BILINEAR) 
                            for f in output_frames_pil]
    
    # Convert back to numpy arrays
    output_frames_np = [np.array(f) for f in output_frames_resized]
    
    return output_frames_np


def load_videomama_pipeline(base_model_path=None, unet_checkpoint_path=None, device="cuda"):
    """
    Load VideoMaMa pipeline with pretrained weights
    
    Args:
        base_model_path: Path to SVD base model (if None, uses default)
        unet_checkpoint_path: Path to VideoMaMa UNet checkpoint (if None, uses default)
        device: Device to run on
        
    Returns:
        VideoInferencePipeline instance
    """
    # Use provided paths or defaults
    if base_model_path is None:
        base_model_path = "checkpoints/stable-video-diffusion-img2vid-xt"
    
    if unet_checkpoint_path is None:
        unet_checkpoint_path = "checkpoints/videomama"
    
    # Check if paths exist
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"SVD base model not found at {base_model_path}. "
            f"Please ensure models are downloaded correctly."
        )
    
    if not os.path.exists(unet_checkpoint_path):
        raise FileNotFoundError(
            f"VideoMaMa checkpoint not found at {unet_checkpoint_path}. "
            f"Please upload your VideoMaMa model to Hugging Face Hub and update the download logic."
        )
    
    print(f"Loading VideoMaMa pipeline...")
    print(f"  Base model: {base_model_path}")
    print(f"  UNet checkpoint: {unet_checkpoint_path}")
    
    pipeline = VideoInferencePipeline(
        base_model_path=base_model_path,
        unet_checkpoint_path=unet_checkpoint_path,
        weight_dtype=torch.float16,
        device=device
    )
    
    print("VideoMaMa pipeline loaded successfully!")
    
    return pipeline
