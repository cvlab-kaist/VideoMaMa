# run_inference.py

import argparse
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from diffusers.utils import check_min_version, export_to_video

# Import the reusable pipeline class from the other file
from pipeline_svd_mask import VideoInferencePipeline

# --- Dependency Check ---
check_min_version("0.24.0.dev0")


# =================================================================================
# Data Loading and Augmentation Helpers
# These functions handle preparing the data before it's sent to the pipeline.
# =================================================================================

def _resize_with_aspect_ratio(image, target_width, target_height):
    """
    Resizes an image maintaining its aspect ratio. The longest side of the image
    is scaled to the maximum of target_width and target_height.
    The shorter side is scaled proportionally.
    """
    max_dim = max(target_width, target_height)
    original_width, original_height = image.size

    if max_dim > min(original_width, original_height):
        return image

    # Determine scaling factor based on the longest side
    if original_width > original_height:
        scale_factor = max_dim / float(original_width)
    else:
        scale_factor = max_dim / float(original_height)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image using the calculated dimensions
    resized_image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    return resized_image


def load_image_sequence(image_folder, mask_folder, num_frames_to_generate, num_frames_to_read, width, height,
                        keep_aspect_ratio=False):
    """
    Loads a sequence of images and corresponding masks from folders.
    """
    all_image_files = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
    if not all_image_files:
        raise ValueError(f"No images found in folder: {image_folder}")

    num_to_read = min(len(all_image_files), num_frames_to_read)
    source_image_files = all_image_files[:num_to_read]

    cond_frames = []
    mask_frames = []

    all_mask_files = os.listdir(mask_folder)
    mask_file_map = {os.path.splitext(f)[0]: f for f in all_mask_files}

    for i in range(num_frames_to_generate):
        file_index = min(i, len(source_image_files) - 1)
        image_filename = source_image_files[file_index]
        image_path = os.path.join(image_folder, image_filename)

        image_name_without_ext = os.path.splitext(image_filename)[0]
        mask_filename = mask_file_map.get(image_name_without_ext)

        if mask_filename is None:
            raise FileNotFoundError(f"Could not find a matching mask file for '{image_filename}' in '{mask_folder}'")

        mask_path = os.path.join(mask_folder, mask_filename)

        cond_image_pil = Image.open(image_path).convert("RGB")
        mask_image_pil = Image.open(mask_path).convert("L")

        if keep_aspect_ratio:
            resized_cond = _resize_with_aspect_ratio(cond_image_pil, width, height)
            resized_mask = _resize_with_aspect_ratio(mask_image_pil, width, height)
        else:
            resized_cond = cond_image_pil.resize((width, height), Image.Resampling.BILINEAR)
            resized_mask = mask_image_pil.resize((width, height), Image.Resampling.BILINEAR)

        cond_frames.append(resized_cond)
        mask_frames.append(resized_mask)

    return cond_frames, mask_frames


def _augment_to_bounding_box(mask_image):
    mask_np = np.array(mask_image)
    points = cv2.findNonZero(mask_np)
    if points is None: return Image.new('L', mask_image.size, 0)
    x, y, w, h = cv2.boundingRect(points)
    new_mask = Image.new('L', mask_image.size, 0)
    draw = ImageDraw.Draw(new_mask)
    draw.rectangle([(x, y), (x + w, y + h)], fill=255)
    return new_mask


def _augment_to_polygon(mask_image, simplification_tolerance):
    """
    Converts all parts of a mask to simplified polygons, preserving all
    disconnected components. The level of simplification is controlled by
    `simplification_tolerance`.
    """
    # Convert the PIL image to a NumPy array for OpenCV processing
    mask_np = np.array(mask_image)

    # Find all external contours in the mask
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return an empty image
    if not contours:
        return Image.new('L', mask_image.size, 0)

    # Create a new blank mask to draw our polygons on
    new_mask = Image.new('L', mask_image.size, 0)
    draw = ImageDraw.Draw(new_mask)

    # ⭐️ KEY CHANGE: Loop through every contour found, not just the largest one
    for contour in contours:
        # Optional: you can ignore very small contours that might be noise
        if cv2.contourArea(contour) < 4:
            continue

        # Calculate the simplification tolerance for each contour individually
        epsilon = simplification_tolerance * cv2.arcLength(contour, True)
        approximated_polygon = cv2.approxPolyDP(contour, epsilon, True)

        # The polygon points need to be in a list of tuples format for drawing
        # We also ensure the polygon has at least 3 vertices to be a valid shape
        if approximated_polygon.shape[0] >= 3:
            # Squeeze the array from (num_points, 1, 2) to (num_points, 2)
            squeezed_points = approximated_polygon.squeeze(axis=1)
            # Convert the NumPy points to a list of tuples
            polygon_points = [tuple(point) for point in squeezed_points]
            # Draw the resulting polygon on the new mask
            draw.polygon(polygon_points, fill=255)

    return new_mask


def _augment_by_resizing(mask_image, downsample_factor):
    original_size = mask_image.size
    small_size = (original_size[0] // downsample_factor, original_size[1] // downsample_factor)
    downsampled = mask_image.resize(small_size, Image.Resampling.BILINEAR)
    upsampled = downsampled.resize(original_size, Image.Resampling.BILINEAR)
    return upsampled.point(lambda p: 255 if p > 127 else 0, mode='L')


def _augment_with_temporal_occlusion(mask_frames, num_occlusions, occlusion_shape, occlusion_scale_range, kernel_size=5, seed=None):
    """
    Applies a diverse set of temporal augmentations to randomly selected mask frames.
    For each frame selected for augmentation, one of the following operations is chosen randomly:
    1. Occlusion: The original method of adding a shape to hide part of the mask.
    2. None Mask: Replaces the mask with a completely empty (black) frame.
    3. All Mask: Replaces the mask with a completely full (white) frame.
    4. Erosion: Erodes the mask boundaries.
    5. Dilation: Dilates the mask boundaries.
    """
    if not mask_frames:
        return mask_frames

    # Initialize a random number generator with the given seed for reproducibility
    rng = random.Random(seed)

    new_mask_frames = list(mask_frames)
    # Ensure we don't try to augment more frames than available
    num_to_augment = min(len(mask_frames), num_occlusions)
    indices_to_augment = rng.sample(range(len(mask_frames)), k=num_to_augment)

    print(f"INFO: Applying temporal augmentation to {len(indices_to_augment)} frames: {indices_to_augment}")

    for idx in indices_to_augment:
        original_mask = new_mask_frames[idx]

        # Define the pool of possible augmentation operations for each frame
        def occlude_mask(mask):
            occluded_mask = mask.copy()
            draw = ImageDraw.Draw(occluded_mask)
            mask_np = np.array(occluded_mask)
            points = cv2.findNonZero(mask_np)
            if points is None: return mask

            x, y, w, h = cv2.boundingRect(points)
            current_scale = rng.uniform(occlusion_scale_range[0], occlusion_scale_range[1])
            occlusion_w, occlusion_h = int(w * current_scale), int(h * current_scale)
            max_offset_x, max_offset_y = w - occlusion_w, h - occlusion_h
            offset_x = rng.randint(0, max_offset_x) if max_offset_x > 0 else 0
            offset_y = rng.randint(0, max_offset_y) if max_offset_y > 0 else 0
            occlusion_x, occlusion_y = x + offset_x, y + offset_y

            if occlusion_shape == 'rectangle':
                draw.rectangle([(occlusion_x, occlusion_y), (occlusion_x + occlusion_w, occlusion_y + occlusion_h)],
                               fill=0)
            elif occlusion_shape == 'circle':
                draw.ellipse([(occlusion_x, occlusion_y), (occlusion_x + occlusion_w, occlusion_y + occlusion_h)],
                             fill=0)
            return occluded_mask

        def none_mask(mask):
            return Image.new('L', mask.size, 0)

        def all_mask(mask):
            return Image.new('L', mask.size, 255)

        def erode_mask(mask):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_np = np.array(mask)
            eroded_np = cv2.erode(mask_np, kernel, iterations=1)
            return Image.fromarray(eroded_np, mode='L')

        def dilate_mask(mask):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_np = np.array(mask)
            dilated_np = cv2.dilate(mask_np, kernel, iterations=1)
            return Image.fromarray(dilated_np, mode='L')

        augmentation_choices = [occlude_mask, none_mask, all_mask, erode_mask, dilate_mask]

        # Randomly select and apply one augmentation to the current frame
        chosen_augmentation = rng.choice(augmentation_choices)
        new_mask_frames[idx] = chosen_augmentation(original_mask)

    return new_mask_frames


# =================================================================================
# Main Execution Logic
# =================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference script using the VideoInferencePipeline.")

    # --- Paths ---
    parser.add_argument("--base_model_path", type=str,
                        default="checkpoints/stabilityai/stable-video-diffusion-img2vid-xt",
                        help="Path to the base SVD model directory.")
    parser.add_argument("--unet_checkpoint_path", type=str, required=True,
                        help="Path to the fine-tuned UNet checkpoint.")
    parser.add_argument("--image_root_path", type=str, required=True,
                        help="Root folder containing input image sequences.")
    parser.add_argument("--mask_root_path", type=str, required=True,
                        help="Root folder containing input mask sequences.")
    parser.add_argument("--output_dir", type=str, default="output_batch", help="Directory to save all outputs.")

    # --- Inference Config ---
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate.")
    parser.add_argument("--num_input_frames", type=int, default=None,
                        help="Number of frames to read from input folders. Defaults to --num_frames.")
    parser.add_argument("--width", type=int, default=1024, help="Processing width.")
    parser.add_argument("--height", type=int, default=576, help="Processing height.")
    parser.add_argument("--keep_aspect_ratio", action="store_true", help="Maintain aspect ratio with padding.")
    parser.add_argument("--mask_cond_mode", type=str, default="vae", choices=["vae", "interpolate"],
                        help="Mask conditioning mode.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help="Mixed precision.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")

    # --- Mask Augmentation and Saving Config ---
    parser.add_argument("--mask_augmentation", type=str, default="none",
                        choices=["none", "polygon", "downsample", 'bounding_box'], help="Mask augmentation type.")
    parser.add_argument("--downsample_factor", type=int, default=8, help="Factor for 'downsample' augmentation.")
    parser.add_argument("--target_mask_points", type=int, default=10, help="Target points for 'polygon' augmentation.")
    parser.add_argument("--save_processed_mask", action="store_true",
                        help="Save the final processed masks used as input to the model.")
    parser.add_argument("--simplification_tolerance", type=float, default=0.001,
                        help="Tolerance for 'polygon' augmentation.")

    parser.add_argument("--temporal_augmentation", action="store_true",
                        help="Apply diverse temporal augmentations to random mask frames.")
    parser.add_argument("--num_occlusions", type=int, default=1,
                        help="Number of frames to apply temporal augmentation to.")
    parser.add_argument("--occlusion_shape", type=str, default="rectangle", choices=["rectangle", "circle"],
                        help="Shape for the temporal occlusion operation.")
    parser.add_argument("--occlusion_scale_range", type=float, nargs=2, default=[0.2, 0.5],
                        help="Range [min, max] for the scale of the occlusion relative to the mask's bounding box.")
    parser.add_argument("--erosion_dilation_kernel_size", type=int, default=5,
                        help="Kernel size for the erosion and dilation operations.")
    parser.add_argument("--input_threshold", type=int, default=127)

    args = parser.parse_args()
    if args.num_input_frames is None:
        args.num_input_frames = args.num_frames

    pipeline = VideoInferencePipeline(
        base_model_path=args.base_model_path,
        unet_checkpoint_path=args.unet_checkpoint_path,
        weight_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    )

    video_folders = sorted(
        [d for d in os.listdir(args.image_root_path) if os.path.isdir(os.path.join(args.image_root_path, d))])

    if not video_folders:
        print(f"Error: No video folders found in '{args.image_root_path}'. Exiting.")
        exit()

    print(f"--- Found {len(video_folders)} videos to process. Starting batch inference. ---")

    for video_name in tqdm(video_folders, desc="Processing Videos"):
        print(f"\n--- Processing: {video_name} ---")

        image_folder_path = os.path.join(args.image_root_path, video_name)
        mask_folder_path = os.path.join(args.mask_root_path, video_name)

        if not os.path.isdir(mask_folder_path):
            print(f"Warning: Mask folder not found for '{video_name}' at '{mask_folder_path}'. Skipping.")
            continue

        try:
            cond_frames, mask_frames = load_image_sequence(
                image_folder_path, mask_folder_path,
                num_frames_to_generate=args.num_frames,
                num_frames_to_read=args.num_input_frames,
                width=args.width, height=args.height,
                keep_aspect_ratio=args.keep_aspect_ratio
            )

            # Apply binary threshold and optional augmentation to masks
            mask_frames = [frame.point(lambda p: 255 if p > args.input_threshold else 0, mode='L') for frame in mask_frames]
            if args.mask_augmentation != "none":
                print(f"Applying '{args.mask_augmentation}' augmentation to masks...")
                if args.mask_augmentation == 'polygon':
                    mask_frames = [_augment_to_polygon(frame, args.simplification_tolerance) for frame in mask_frames]
                elif args.mask_augmentation == 'downsample':
                    mask_frames = [_augment_by_resizing(frame, args.downsample_factor) for frame in mask_frames]
                elif args.mask_augmentation == 'bounding_box':
                    mask_frames = [_augment_to_bounding_box(frame) for frame in mask_frames]

            if args.temporal_augmentation:
                mask_frames = _augment_with_temporal_occlusion(
                    mask_frames,
                    num_occlusions=args.num_occlusions,
                    occlusion_shape=args.occlusion_shape,
                    occlusion_scale_range=args.occlusion_scale_range,
                    kernel_size=args.erosion_dilation_kernel_size,
                    seed=video_name
                )

            # Save the processed masks if the flag is set
            if args.save_processed_mask:
                mask_save_folder = os.path.join(args.output_dir, "mask_guide", video_name)
                os.makedirs(mask_save_folder, exist_ok=True)
                print(f"Saving {len(mask_frames)} processed masks to: {mask_save_folder}")
                for i, frame in enumerate(mask_frames):
                    frame.save(os.path.join(mask_save_folder, f"frame_{i:04d}.png"))

            print("Running inference...")
            generated_frames = pipeline.run(
                cond_frames=cond_frames,
                mask_frames=mask_frames,
                seed=args.seed,
                mask_cond_mode=args.mask_cond_mode
            )

            results_folder = os.path.join(args.output_dir, "results", video_name)
            os.makedirs(results_folder, exist_ok=True)

            print(f"Saving {len(generated_frames)} generated frames to: {results_folder}")
            for i, frame in enumerate(generated_frames):
                frame.save(os.path.join(results_folder, f"frame_{i:04d}.png"))

            if len(generated_frames) > 1:
                video_save_path = os.path.join(results_folder, "video.mp4")
                export_to_video(generated_frames, video_save_path, fps=7)
                print(f"Video saved to {video_save_path}")

        except Exception as e:
            print(f"\nAn error occurred while processing {video_name}: {e}")
            print("Skipping to the next video.")
            continue

    print("\nBatch inference complete.")