from src.s3_utils import download_file
from torch.utils.data import RandomSampler, Dataset, Subset
import pandas as pd
import io
import os
import warnings
from PIL import Image, ImageDraw
import random
import torchvision
import torch
from torchvision import transforms

import numpy as np
import cv2

from dataloader.augmentations import (
    augment_to_bounding_box,
    augment_to_polygon,
    augment_by_resizing,
    apply_all_augmentations,
    augment_with_temporal_occlusion,
    augment_to_polygon_preserve_all_parts,
    augment_with_instability,
    augment_to_polygon_with_nested_holes
)


class AdobeVideoDataset(Dataset):
    """
    Dataset for loading video data stored as image sequences on S3.
    Includes functionality to generate a binary mask from the alpha channel
    and apply augmentations to simplify the mask's details.
    """

    def __init__(
            self,
            s3_bucket,
            s3_prefix,
            s3_metadata_key,
            num_frames=16,
            height=512,
            width=512,
            data_file_keys=("video", "alpha", "binary_mask"),
            repeat=1,
            binary_mask_threshold=127,
            mask_augmentation="none",  # 'polygon', 'downsample', 'bounding_box', 'instability', or 'all'
            downsample_factors=[8, 16, 32],
            simplification_tolerance=0.005,
            # --- NEW: Instability Augmentation Parameters ---
            instability_noise_level=0.05,
            # --- UPDATED: Temporal Augmentation Parameters ---
            temporal_augmentation_rate=0.5,
            num_occlusions=1,
            occlusion_shape='rectangle',
            occlusion_scale_range=(0.3, 0.6),
            augmentation_ratios=None,  # Ratio for applying augmentations
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_metadata_key = s3_metadata_key
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.data_file_keys = data_file_keys
        self.repeat = repeat
        self.binary_mask_threshold = binary_mask_threshold
        self.mask_augmentation = mask_augmentation

        if isinstance(downsample_factors, int):
            self.downsample_factors = [downsample_factors]
        else:
            self.downsample_factors = downsample_factors

        self.simplification_tolerance = simplification_tolerance

        # --- NEW: Store instability parameter ---
        self.instability_noise_level = instability_noise_level

        self.augmentation_ratios = augmentation_ratios

        # --- UPDATED: Store temporal augmentation parameters ---
        self.temporal_augmentation_rate = temporal_augmentation_rate
        self.num_occlusions = num_occlusions
        self.occlusion_shape = occlusion_shape
        self.occlusion_scale_range = occlusion_scale_range
        # --------------------------------------------------

        # ⭐️ UPDATE: Add 'instability' to the validation check
        if self.mask_augmentation not in ["none", 'polygon', 'downsample', 'all', 'bounding_box', 'instability']:
            raise ValueError(
                "mask_augmentation must be one of 'polygon', 'downsample', 'all', 'bounding_box', 'instability', or 'none'")

        print(f"Loading metadata from s3://{self.s3_bucket}/{self.s3_metadata_key}")
        try:
            parquet_data = download_file(self.s3_bucket, self.s3_metadata_key)
            self.df = pd.read_parquet(io.BytesIO(parquet_data))
            print(f"Successfully loaded {len(self.df)} records from metadata.")
            if self.mask_augmentation != "none":
                print(f"Applying '{self.mask_augmentation}' augmentation to binary masks.")
                if 'downsample' in self.mask_augmentation or 'all' in self.mask_augmentation:
                    print(f"Using random downsample factors from: {self.downsample_factors}")
                if self.mask_augmentation == 'all' and self.augmentation_ratios:
                    print(f"with ratios: {self.augmentation_ratios}")
            if self.temporal_augmentation_rate > 0:
                print(
                    f"Applying temporal occlusion augmentation with a {self.temporal_augmentation_rate:.0%} probability.")
        except Exception as e:
            print(f"FATAL: Error loading metadata from S3: {e}")
            self.df = pd.DataFrame()

    def __len__(self):
        return len(self.df) * self.repeat

    def _load_frames_from_s3(self, frame_dir_s3_key, frame_indices, key_type):
        clip_images = []
        is_alpha = key_type in ['alpha', 'binary_mask']
        extension = '.png' if is_alpha else '.jpg'
        mode = 'L' if is_alpha else 'RGB'

        if key_type == 'binary_mask':
            key_type = 'alpha'

        for frame_idx in frame_indices:
            frame_s3_key = os.path.join(frame_dir_s3_key, f'{frame_idx:04d}{extension}')
            full_s3_path = os.path.join(self.s3_prefix, frame_s3_key)

            try:
                image_data = download_file(self.s3_bucket, full_s3_path)
                image = Image.open(io.BytesIO(image_data)).convert(mode)
                clip_images.append(image)
            except Exception as e:
                warnings.warn(f"Could not load frame {full_s3_path}. Error: {e}")
                return None
        return clip_images

    def crop_and_resize(self, image, target_height, target_width):
        w, h = image.size
        scale = max(target_width / w, target_height / h)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = torchvision.transforms.functional.resize(
            image,
            (target_height, target_width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def __getitem__(self, idx):
        random.seed(idx)

        record_idx = idx % len(self.df)
        record = self.df.iloc[record_idx]

        total_frames_in_video = int(record['num_frames'])
        if total_frames_in_video < 1:
            raise ValueError(f"Video for record {record_idx} is empty.")

        if total_frames_in_video >= self.num_frames:
            start_frame = random.randint(0, total_frames_in_video - self.num_frames)
            frame_indices = list(range(start_frame, start_frame + self.num_frames))
        else:
            base_indices = list(range(total_frames_in_video))
            if total_frames_in_video > 1:
                base_indices.extend(list(range(total_frames_in_video - 2, 0, -1)))
            num_repeats = (self.num_frames + len(base_indices) - 1) // len(base_indices)
            repeated_indices = base_indices * num_repeats
            frame_indices = repeated_indices[:self.num_frames]

        output_data = {}
        alpha_pil_frames = None

        for key in self.data_file_keys:
            if key == 'binary_mask':
                continue

            column_map = {'video': 'composite_path', 'alpha': 'alpha_path', 'fg': 'fg_path', 'bg': 'bg_path'}
            column_name = column_map.get(key, key)

            if column_name not in record or pd.isna(record[column_name]):
                continue

            frame_dir_path = record[column_name]
            frames = self._load_frames_from_s3(frame_dir_path, frame_indices, key_type=key)

            if frames is None:
                warnings.warn(f"Failed to load frames for key '{key}' in record {record_idx}. Retrying with next item.")
                return self.__getitem__((idx + 1))

            if key == 'alpha':
                alpha_pil_frames = frames

            processed_frames = [self.crop_and_resize(img, self.height, self.width) for img in frames]
            tensor_frames = torch.stack([transforms.ToTensor()(img) for img in processed_frames])
            tensor_frames = tensor_frames * 2.0 - 1.0
            output_data[key] = tensor_frames

        if 'binary_mask' in self.data_file_keys and alpha_pil_frames is not None:
            binary_mask_frames = [
                alpha_frame.point(lambda p: 255 if p > self.binary_mask_threshold else 0, mode='L')
                for alpha_frame in alpha_pil_frames
            ]

            chosen_downsample_factor = random.choice(self.downsample_factors)

            if self.mask_augmentation == 'polygon':
                binary_mask_frames = [augment_to_polygon_with_nested_holes(frame, self.simplification_tolerance) for
                                      frame in
                                      binary_mask_frames]
            elif self.mask_augmentation == 'downsample':
                binary_mask_frames = [augment_by_resizing(frame, chosen_downsample_factor) for frame in
                                      binary_mask_frames]
            elif self.mask_augmentation == 'bounding_box':
                binary_mask_frames = [augment_to_bounding_box(frame) for frame in binary_mask_frames]

            # --- ⭐️ NEW: Add logic to apply instability augmentation ---
            elif self.mask_augmentation == 'instability':
                binary_mask_frames = [augment_with_instability(frame, self.instability_noise_level) for frame in
                                      binary_mask_frames]

            elif self.mask_augmentation == 'all':
                binary_mask_frames = [
                    # ⭐️ PASS THE NEW PARAMETER
                    apply_all_augmentations(frame, chosen_downsample_factor, self.simplification_tolerance,
                                            self.instability_noise_level)
                    for frame in binary_mask_frames]

            # Apply temporal occlusion
            if random.random() < self.temporal_augmentation_rate:
                binary_mask_frames = augment_with_temporal_occlusion(
                    binary_mask_frames, self.num_occlusions, self.occlusion_shape, self.occlusion_scale_range
                )

            processed_binary_frames = [self.crop_and_resize(img, self.height, self.width) for img in
                                       binary_mask_frames]

            tensor_frames = torch.stack([transforms.ToTensor()(img) for img in processed_binary_frames])
            tensor_frames = tensor_frames * 2.0 - 1.0

            output_data['binary_mask'] = tensor_frames

        return output_data