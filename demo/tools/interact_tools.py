"""
SAM2 Interaction Tools
Handles SAM2 mask generation with user clicks
"""

import sys
sys.path.append("/home/cvlab19/project/samuel/CVPR/sam2")

import numpy as np
from PIL import Image
from .base_segmenter import BaseSegmenter
from .painter import mask_painter, point_painter


mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8  # positive points
point_color_ps = 50 # negative points
point_alpha = 0.9
point_radius = 15


class SamControler:
    def __init__(self, SAM_checkpoint, model_type, device):
        """
        Initialize SAM controller
        
        Args:
            SAM_checkpoint: Path to SAM2 checkpoint
            model_type: SAM2 model config file
            device: Device to run on
        """
        self.sam_controler = BaseSegmenter(SAM_checkpoint, model_type, device)
        self.device = device
    
    def first_frame_click(self, image: np.ndarray, points: np.ndarray, 
                         labels: np.ndarray, multimask=True, mask_color=3):
        """
        Generate mask from clicks on first frame
        
        Args:
            image: np.ndarray, (H, W, 3), RGB image
            points: np.ndarray, (N, 2), [x, y] coordinates
            labels: np.ndarray, (N,), 1 for positive, 0 for negative
            multimask: bool, whether to generate multiple masks
            mask_color: int, color ID for mask overlay
            
        Returns:
            mask: np.ndarray, (H, W), binary mask
            logit: np.ndarray, (H, W), mask logits
            painted_image: PIL.Image, visualization with mask and points
        """
        # Check if we have positive clicks
        neg_flag = labels[-1]
        
        if neg_flag == 1:  # Has positive click
            # First pass with points only
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            
            # Refine with mask input
            prompts = {
                'point_coords': points,
                'point_labels': labels,
                'mask_input': logit[None, :, :]
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'both', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        else:  # Only positive clicks
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        
        # Paint mask on image
        painted_image = mask_painter(
            image, 
            mask.astype('uint8'), 
            mask_color, 
            mask_alpha, 
            contour_color, 
            contour_width
        )
        
        # Paint positive points (label > 0)
        positive_points = np.squeeze(points[np.argwhere(labels > 0)], axis=1)
        if len(positive_points) > 0:
            painted_image = point_painter(
                painted_image, 
                positive_points, 
                point_color_ne, 
                point_alpha, 
                point_radius, 
                contour_color, 
                contour_width
            )
        
        # Paint negative points (label < 1)
        negative_points = np.squeeze(points[np.argwhere(labels < 1)], axis=1)
        if len(negative_points) > 0:
            painted_image = point_painter(
                painted_image, 
                negative_points, 
                point_color_ps, 
                point_alpha, 
                point_radius, 
                contour_color, 
                contour_width
            )
        
        painted_image = Image.fromarray(painted_image)
        
        return mask, logit, painted_image
