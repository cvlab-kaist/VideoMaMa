"""
SAM2 Wrapper for Video Mask Tracking - Hugging Face Space Version
Handles mask generation and propagation through video
"""

import sys
import os
from pathlib import Path

# Add SAM2 to path if installed
try:
    import sam2
except ImportError:
    # Try to add from common locations
    possible_paths = [
        "/home/cvlab19/project/samuel/CVPR/sam2",
        "./sam2"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.append(path)
            break

import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
import tempfile
import shutil

from sam2.build_sam import build_sam2_video_predictor


class SAM2VideoTracker:
    def __init__(self, checkpoint_path, config_file, device="cuda"):
        """
        Initialize SAM2 video tracker
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint
            config_file: Path to SAM2 config file
            device: Device to run on
        """
        self.device = device
        self.predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=device
        )
        print(f"SAM2 video tracker initialized on {device}")
    
    def track_video(self, frames: List[np.ndarray], points: List[List[int]], 
                   labels: List[int]) -> List[np.ndarray]:
        """
        Track object through video using SAM2
        
        Args:
            frames: List of numpy arrays, [(H,W,3)]*n, uint8 RGB frames
            points: List of [x, y] coordinates for prompts
            labels: List of labels (1 for positive, 0 for negative)
            
        Returns:
            masks: List of numpy arrays, [(H,W)]*n, uint8 binary masks
        """
        # Create temporary directory for frames
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        try:
            # Save frames to temp directory
            print(f"Saving {len(frames)} frames to temporary directory...")
            for i, frame in enumerate(frames):
                frame_path = frames_dir / f"{i:05d}.jpg"
                Image.fromarray(frame).save(frame_path, quality=95)
            
            # Initialize SAM2 video predictor
            print("Initializing SAM2 inference state...")
            inference_state = self.predictor.init_state(video_path=str(frames_dir))
            
            # Add prompts on first frame
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            
            print(f"Adding {len(points)} point prompts on first frame...")
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_array,
                labels=labels_array,
            )
            
            # Propagate through video
            print("Propagating masks through video...")
            masks = []
            for frame_idx, object_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
                # Get mask for object ID 1
                obj_ids_list = object_ids.tolist() if hasattr(object_ids, 'tolist') else object_ids
                
                if 1 in obj_ids_list:
                    mask_idx = obj_ids_list.index(1)
                    mask = (mask_logits[mask_idx] > 0.0).cpu().numpy()
                    mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                    masks.append(mask_uint8)
                else:
                    # No mask for this frame, use empty mask
                    h, w = frames[0].shape[:2]
                    masks.append(np.zeros((h, w), dtype=np.uint8))
            
            print(f"Generated {len(masks)} masks")
            return masks
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def get_first_frame_mask(self, frame: np.ndarray, points: List[List[int]], 
                            labels: List[int]) -> np.ndarray:
        """
        Get mask for first frame only (for preview)
        
        Args:
            frame: np.ndarray, (H, W, 3), uint8 RGB frame
            points: List of [x, y] coordinates
            labels: List of labels (1 for positive, 0 for negative)
            
        Returns:
            mask: np.ndarray, (H, W), uint8 binary mask
        """
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        try:
            # Save single frame
            frame_path = frames_dir / "00000.jpg"
            Image.fromarray(frame).save(frame_path, quality=95)
            
            # Initialize SAM2
            inference_state = self.predictor.init_state(video_path=str(frames_dir))
            
            # Add prompts
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_array,
                labels=labels_array,
            )
            
            # Get mask
            if len(out_mask_logits) > 0:
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
                return mask_uint8
            else:
                return np.zeros(frame.shape[:2], dtype=np.uint8)
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_sam2_tracker(checkpoint_path=None, device="cuda"):
    """
    Load SAM2 video tracker with pretrained weights
    
    Args:
        checkpoint_path: Path to SAM2 checkpoint (if None, uses default location)
        device: Device to run on
        
    Returns:
        SAM2VideoTracker instance
    """
    # Use provided path or default
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    
    # Config file should be in the SAM2 repo
    config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # Check if we need to use the local yaml file
    if not os.path.exists(config_file):
        config_file = "sam2_hiera_l.yaml"
    
    print(f"Loading SAM2 from {checkpoint_path}...")
    print(f"Using config: {config_file}")
    
    tracker = SAM2VideoTracker(checkpoint_path, config_file, device)
    
    return tracker
