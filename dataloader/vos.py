# dataloader.py

import os
import random
from PIL import Image
from torch.utils.data import Dataset
import warnings
import numpy as np
from torchvision.transforms import functional as TF
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
from torchvision.transforms import InterpolationMode



import numpy as np
import cv2

# Assuming transforms.py is in the same directory or accessible
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop
from dataloader.augmentations import (
    augment_to_bounding_box,
    augment_to_polygon,
    augment_by_resizing,
    apply_all_augmentations,
    augment_with_temporal_occlusion
)


class VideoObjectSegmentationDataset(Dataset):
    def __init__(self, root_path, num_frames=16, height=576, width=1024,
                 mask_augmentation="none", simplification_tolerance=0.005, downsample_factor=8,
                 temporal_augmentation_rate=0.5, num_occlusions=1, occlusion_shape='rectangle',
                 occlusion_scale_range=(0.3, 0.6)):
        self.root_path = root_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_augmentation = mask_augmentation
        self.simplification_tolerance = simplification_tolerance
        self.downsample_factor = downsample_factor
        self.temporal_augmentation_rate = temporal_augmentation_rate
        self.num_occlusions = num_occlusions
        self.occlusion_shape = occlusion_shape
        self.occlusion_scale_range = occlusion_scale_range

        self.image_dir = os.path.join(root_path, "JPEGImages")
        self.mask_dir = os.path.join(root_path, "Annotations")
        self.videos = sorted([v for v in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.mask_dir, v))])
        print(f"Found {len(self.videos)} video sequences in {root_path}")

    def __len__(self):
        return len(self.videos)

    # # --- Augmentation methods (adapted from AdobeVideoDataset) ---
    # def _augment_to_bounding_box(self, mask_image):
    #     mask_np = np.array(mask_image)
    #     points = cv2.findNonZero(mask_np)
    #     if points is None:
    #         return Image.new('L', mask_image.size, 0)
    #     x, y, w, h = cv2.boundingRect(points)
    #     new_mask = Image.new('L', mask_image.size, 0)
    #     draw = ImageDraw.Draw(new_mask)
    #     draw.rectangle([(x, y), (x + w, y + h)], fill=255)
    #     return new_mask
    #
    # def _augment_to_polygon(self, mask_image):
    #     mask_np = np.array(mask_image)
    #     contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         return Image.new('L', mask_image.size, 0)
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     epsilon = self.simplification_tolerance * cv2.arcLength(largest_contour, True)
    #     approximated_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    #     new_mask = Image.new('L', mask_image.size, 0)
    #     draw = ImageDraw.Draw(new_mask)
    #     squeezed_points = approximated_polygon.squeeze()
    #     if squeezed_points.ndim == 1:
    #         polygon_points = [tuple(squeezed_points)]
    #     else:
    #         polygon_points = [tuple(point) for point in squeezed_points]
    #     if len(polygon_points) > 1:
    #         draw.polygon(polygon_points, fill=255)
    #     return new_mask
    #
    # def _augment_by_resizing(self, mask_image):
    #     original_size = mask_image.size
    #     small_size = (original_size[0] // self.downsample_factor, original_size[1] // self.downsample_factor)
    #     downsampled = mask_image.resize(small_size, Image.Resampling.BILINEAR)
    #     upsampled = downsampled.resize(original_size, Image.Resampling.BILINEAR)
    #     return upsampled.point(lambda p: 255 if p > 127 else 0, mode='L')
    #
    # def _apply_all_augmentations(self, mask_image):
    #     augmentations = [
    #         self._augment_to_polygon,
    #         self._augment_by_resizing,
    #     ]
    #     random.shuffle(augmentations)
    #     num_to_apply = random.randint(1, len(augmentations))
    #     augmented_mask = mask_image
    #     for i in range(num_to_apply):
    #         augmented_mask = augmentations[i](augmented_mask)
    #     return augmented_mask
    #
    # def _augment_with_temporal_occlusion(self, mask_frames):
    #     """
    #     Applies a specified number of occlusions to randomly selected mask frames.
    #     The scale of each occlusion is randomized within a given range.
    #     """
    #     if not mask_frames:
    #         return mask_frames
    #
    #     indices_to_occlude = random.choices(range(len(mask_frames)), k=self.num_occlusions)
    #     new_mask_frames = list(mask_frames)
    #
    #     for idx in indices_to_occlude:
    #         occluded_mask = new_mask_frames[idx].copy()
    #         draw = ImageDraw.Draw(occluded_mask)
    #         mask_np = np.array(occluded_mask)
    #         points = cv2.findNonZero(mask_np)
    #
    #         if points is None:
    #             continue
    #
    #         x, y, w, h = cv2.boundingRect(points)
    #         current_scale = random.uniform(self.occlusion_scale_range[0], self.occlusion_scale_range[1])
    #         occlusion_w = int(w * current_scale)
    #         occlusion_h = int(h * current_scale)
    #         max_offset_x = w - occlusion_w
    #         max_offset_y = h - occlusion_h
    #         offset_x = random.randint(0, max_offset_x) if max_offset_x > 0 else 0
    #         offset_y = random.randint(0, max_offset_y) if max_offset_y > 0 else 0
    #         occlusion_x = x + offset_x
    #         occlusion_y = y + offset_y
    #
    #         if self.occlusion_shape == 'rectangle':
    #             draw.rectangle(
    #                 [(occlusion_x, occlusion_y), (occlusion_x + occlusion_w, occlusion_y + occlusion_h)],
    #                 fill=0
    #             )
    #         elif self.occlusion_shape == 'circle':
    #             draw.ellipse(
    #                 [(occlusion_x, occlusion_y), (occlusion_x + occlusion_w, occlusion_y + occlusion_h)],
    #                 fill=0
    #             )
    #         new_mask_frames[idx] = occluded_mask
    #     return new_mask_frames

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        image_folder = os.path.join(self.image_dir, video_name)
        mask_folder = os.path.join(self.mask_dir, video_name)
        frame_names = sorted(os.listdir(image_folder))
        total_frames = len(frame_names)

        if total_frames < 1:
            warnings.warn(f"Video '{video_name}' is empty. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        if total_frames >= self.num_frames:
            # --- MODIFICATION START ---
            # Always start from the first frame for VOS tasks.
            start_frame = 0
            # --- MODIFICATION END ---
            frame_indices = list(range(start_frame, start_frame + self.num_frames))
        else:
            base_indices = list(range(total_frames))
            if total_frames > 1:
                base_indices.extend(list(range(total_frames - 2, 0, -1)))
            num_repeats = (self.num_frames + len(base_indices) - 1) // len(base_indices)
            repeated_indices = base_indices * num_repeats
            frame_indices = repeated_indices[:self.num_frames]

        selected_frame_names = [frame_names[i] for i in frame_indices]

        first_mask_path = os.path.join(mask_folder, os.path.splitext(selected_frame_names[0])[0] + '.png')
        target_object_id = None
        if os.path.exists(first_mask_path):
            first_mask_img = Image.open(first_mask_path).convert("P")
            mask_array = np.array(first_mask_img)
            object_ids = np.unique(mask_array)
            object_ids = object_ids[object_ids != 0]
            if object_ids.size > 0:
                target_object_id = random.choice(object_ids)

        image_frames, gt_mask_frames = [], []
        for frame_name in selected_frame_names:
            img_path = os.path.join(image_folder, frame_name)
            image = Image.open(img_path).convert("RGB")
            image_frames.append(image)

            mask_binary = None
            if target_object_id is not None:
                mask_path = os.path.join(mask_folder, os.path.splitext(frame_name)[0] + '.png')
                if os.path.exists(mask_path):
                    mask_palette = Image.open(mask_path).convert("P")
                    mask_np = np.array(mask_palette)
                    binary_mask_np = (mask_np == target_object_id).astype(np.uint8) * 255
                    mask_binary = Image.fromarray(binary_mask_np, mode='L')

            if mask_binary is None:
                mask_binary = Image.new('L', image.size, 0)

            gt_mask_frames.append(mask_binary)

        # 1. Create a copy for the input binary mask and apply augmentations
        binary_mask_frames = [m.copy() for m in gt_mask_frames]

        if self.mask_augmentation == 'polygon':
            binary_mask_frames = [augment_to_polygon(frame, self.simplification_tolerance) for frame in
                                  binary_mask_frames]
        elif self.mask_augmentation == 'downsample':
            binary_mask_frames = [augment_by_resizing(frame, self.downsample_factor) for frame in binary_mask_frames]
        elif self.mask_augmentation == 'bounding_box':
            binary_mask_frames = [augment_to_bounding_box(frame) for frame in binary_mask_frames]
        elif self.mask_augmentation == 'all':
            binary_mask_frames = [apply_all_augmentations(frame, self.downsample_factor, self.simplification_tolerance)
                                  for frame in binary_mask_frames]

        # Apply temporal occlusion
        if random.random() < self.temporal_augmentation_rate:
            binary_mask_frames = augment_with_temporal_occlusion(
                binary_mask_frames, self.num_occlusions, self.occlusion_shape, self.occlusion_scale_range
            )

        # 2. Apply synchronized geometric transformations
        # Resize
        image_frames = [self.crop_and_resize(img, self.height, self.width) for img in image_frames]
        gt_mask_frames = [self.crop_and_resize(mask, self.height, self.width) for mask
                          in gt_mask_frames]
        binary_mask_frames = [self.crop_and_resize(mask, self.height, self.width) for
                              mask in binary_mask_frames]

        # Random Horizontal Flip
        if random.random() < 0.5:
            image_frames = [TF.hflip(img) for img in image_frames]
            gt_mask_frames = [TF.hflip(mask) for mask in gt_mask_frames]
            binary_mask_frames = [TF.hflip(mask) for mask in binary_mask_frames]

        # 3. Convert to Tensors
        video_tensor = torch.stack([TF.to_tensor(img) for img in image_frames])
        alpha_tensor = torch.stack([TF.to_tensor(mask) for mask in gt_mask_frames])
        binary_mask_tensor = torch.stack([TF.to_tensor(mask) for mask in binary_mask_frames])

        video_tensor = video_tensor * 2.0 - 1.0
        alpha_tensor = alpha_tensor * 2.0 - 1.0
        binary_mask_tensor = binary_mask_tensor * 2.0 - 1.0

        return {'video': video_tensor, 'alpha': alpha_tensor, 'binary_mask': binary_mask_tensor}

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


# --- Example Usage (Unchanged) ---
if __name__ == '__main__':
    dataset_root = './mose/mose_vos_train'

    # Note: The 'transforms' argument is not used by this Dataset class, so it has been removed.
    # The class handles its own transformations internally.
    train_transforms = Compose([
        Resize((256, 455)),
        RandomCrop((224, 224)),
        RandomHorizontalFlip(p=0.5),
        ToTensor()
    ])

    vos_dataset = VideoObjectSegmentationDataset(
        root_path=dataset_root,
        num_frames=8
    )

    if len(vos_dataset) > 0:
        # The 'mask' key does not exist in the returned dictionary.
        # Let's access 'alpha' (ground truth mask) and 'binary_mask' (augmented input mask).
        sample = vos_dataset[0]
        video_tensor = sample['video']
        alpha_mask_tensor = sample['alpha']
        binary_mask_tensor = sample['binary_mask']


        print("Successfully loaded one sample.")
        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Alpha (GT mask) tensor shape: {alpha_mask_tensor.shape}")
        print(f"Binary mask tensor shape: {binary_mask_tensor.shape}")
        print(f"Alpha mask tensor unique values: {torch.unique(alpha_mask_tensor)}")
    else:
        print("Dataset could not be loaded or is empty.")