"""
VideoMaMa Inference Wrapper
Handles video matting with mask conditioning
"""

import sys
sys.path.append("../")
sys.path.append("../../")

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
import tqdm

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


def load_videomama_pipeline(device="cuda"):
    """
    Load VideoMaMa pipeline with pretrained weights
    
    Args:
        device: Device to run on
        
    Returns:
        VideoInferencePipeline instance
    """
    # Local paths for testing
    base_model_path = "/home/cvlab19/project/samuel/data/CVPR/pretrained_models/stable-video-diffusion-img2vid-xt"
    unet_checkpoint_path = "/home/cvlab19/project/samuel/data/CVPR/pretrained_models/videomama"
    
    print(f"Loading VideoMaMa pipeline from {unet_checkpoint_path}...")
    
    pipeline = VideoInferencePipeline(
        base_model_path=base_model_path,
        unet_checkpoint_path=unet_checkpoint_path,
        weight_dtype=torch.float16,
        device=device
    )
    
    print("VideoMaMa pipeline loaded successfully!")
    
    return pipeline
