"""
Mask and point painting utilities
Adapted from MatAnyone demo
"""

import cv2
import numpy as np
from PIL import Image


def mask_painter(input_image, input_mask, mask_color=5, mask_alpha=0.7, 
                 contour_color=1, contour_width=5):
    """
    Paint mask on image with transparency
    
    Args:
        input_image: np.ndarray, (H, W, 3)
        input_mask: np.ndarray, (H, W), binary mask
        mask_color: int, color ID for mask
        mask_alpha: float, transparency
        contour_color: int, color ID for contour
        contour_width: int, width of contour
        
    Returns:
        painted_image: np.ndarray, (H, W, 3)
    """
    assert input_image.shape[:2] == input_mask.shape, "Image and mask must have same dimensions"
    
    # Color palette
    palette = np.array([
        [0, 0, 0],        # 0: black
        [255, 0, 0],      # 1: red
        [0, 255, 0],      # 2: green
        [0, 0, 255],      # 3: blue
        [255, 255, 0],    # 4: yellow
        [255, 0, 255],    # 5: magenta
        [0, 255, 255],    # 6: cyan
        [128, 128, 128],  # 7: gray
        [255, 165, 0],    # 8: orange
        [128, 0, 128],    # 9: purple
    ])
    
    mask_color_rgb = palette[mask_color % len(palette)]
    contour_color_rgb = palette[contour_color % len(palette)]
    
    # Create colored mask
    painted_image = input_image.copy()
    colored_mask = np.zeros_like(input_image)
    colored_mask[input_mask > 0] = mask_color_rgb
    
    # Blend with alpha
    mask_region = input_mask > 0
    painted_image[mask_region] = (
        painted_image[mask_region] * (1 - mask_alpha) + 
        colored_mask[mask_region] * mask_alpha
    ).astype(np.uint8)
    
    # Draw contour
    if contour_width > 0:
        contours, _ = cv2.findContours(
            input_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            painted_image, 
            contours, 
            -1, 
            contour_color_rgb.tolist(), 
            contour_width
        )
    
    return painted_image


def point_painter(input_image, input_points, point_color=8, point_alpha=0.9,
                  point_radius=15, contour_color=2, contour_width=3):
    """
    Paint points on image
    
    Args:
        input_image: np.ndarray, (H, W, 3)
        input_points: np.ndarray, (N, 2), [x, y] coordinates
        point_color: int, color ID for points
        point_alpha: float, transparency
        point_radius: int, radius of point circles
        contour_color: int, color ID for contour
        contour_width: int, width of contour
        
    Returns:
        painted_image: np.ndarray, (H, W, 3)
    """
    if len(input_points) == 0:
        return input_image
    
    palette = np.array([
        [0, 0, 0],        # 0: black
        [255, 0, 0],      # 1: red
        [0, 255, 0],      # 2: green
        [0, 0, 255],      # 3: blue
        [255, 255, 0],    # 4: yellow
        [255, 0, 255],    # 5: magenta
        [0, 255, 255],    # 6: cyan
        [128, 128, 128],  # 7: gray
        [255, 165, 0],    # 8: orange
        [128, 0, 128],    # 9: purple
    ])
    
    point_color_rgb = palette[point_color % len(palette)]
    contour_color_rgb = palette[contour_color % len(palette)]
    
    painted_image = input_image.copy()
    
    for point in input_points:
        x, y = int(point[0]), int(point[1])
        
        # Draw filled circle with alpha blending
        overlay = painted_image.copy()
        cv2.circle(overlay, (x, y), point_radius, point_color_rgb.tolist(), -1)
        cv2.addWeighted(overlay, point_alpha, painted_image, 1 - point_alpha, 0, painted_image)
        
        # Draw contour
        if contour_width > 0:
            cv2.circle(painted_image, (x, y), point_radius, contour_color_rgb.tolist(), contour_width)
    
    return painted_image
