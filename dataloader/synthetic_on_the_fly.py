import os
import cv2
import yaml
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from collections import Counter

# =====================================================================================
# Helper Functions (from generate_synthetic_v2.py and synthetic.py)
# =====================================================================================

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


def get_files_from_folder(folder_path):
    """Helper function to get sorted image files from a directory."""
    if not os.path.isdir(folder_path): return []
    files = []
    for f in os.listdir(folder_path):
        if isinstance(f, str) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            files.append(os.path.join(folder_path, f))
    return sorted(files)


# =====================================================================================
# On-the-Fly Dataloader Class
# =====================================================================================

class OnTheFlySyntheticDataset(Dataset):
    """
    A PyTorch Dataset that generates synthetic video data on-the-fly.
    It merges the logic of a generator script with a dataloader, avoiding
    the need to save the dataset to disk.
    """

    def __init__(
            self,
            config_file,
            epoch_size=1000,
            num_frames=16,
            height=512,
            width=512,
            min_instances=1,
            max_instances=3,
            fg_scale_range=(0.5, 0.99),
            rotation_speed=1.25,
            zoom_range=(0.8, 1.2),
            translation_range=(0.1, 0.2),
            binary_mask_threshold=127,
            mask_augmentation="none",
            downsample_factors=(8, 16),
            simplification_tolerance=0.005,
            instability_noise_level=0.05,
            temporal_augmentation_rate=0.5,
            num_occlusions=1,
            occlusion_shape='rectangle',
            occlusion_scale_range=(0.3, 0.6),
    ):
        """
        Initializes the on-the-fly dataset generator.

        Args:
            config_file (str): Path to the YAML config file specifying fg/bg sources.
            epoch_size (int): The number of unique videos to generate per epoch.
            num_frames (int): The number of frames to generate per clip.
            height (int): The output height of the frames.
            width (int): The output width of the frames.
            ... and other generation/augmentation parameters.
        """
        self.epoch_size = epoch_size
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.output_resolution = (width, height)

        # --- Generation parameters ---
        self.min_instances_arg = min_instances
        self.max_instances_arg = max_instances
        self.fg_scale_range_arg = fg_scale_range
        self.rotation_speed_arg = rotation_speed
        self.zoom_range_arg = zoom_range
        self.translation_range_arg = translation_range

        # --- Augmentation parameters ---
        self.binary_mask_threshold = binary_mask_threshold
        self.mask_augmentation = mask_augmentation
        self.downsample_factors = downsample_factors
        self.simplification_tolerance = simplification_tolerance
        self.instability_noise_level = instability_noise_level
        self.temporal_augmentation_rate = temporal_augmentation_rate
        self.num_occlusions = num_occlusions
        self.occlusion_shape = occlusion_shape
        self.occlusion_scale_range = occlusion_scale_range

        # --- Load sources from config file (logic from generate_synthetic_v2.py) ---
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self._load_sources(config)

        # --- Transformation for output tensors ---
        self.to_tensor = transforms.ToTensor()

    def _load_sources(self, config):
        """Scans directories listed in the config file to find fg/bg assets."""
        self.fg_sources = []
        for source_def in config.get('foreground_sources', []):
            source_type = source_def.get('type')
            fg_parent = source_def.get('fg_path')
            alpha_parent = source_def.get('alpha_path')

            if not all([source_type, fg_parent, alpha_parent]):
                print(f"Skipping incomplete source definition: {source_def}")
                continue

            if source_type == 'video':
                for item in os.listdir(fg_parent):
                    if os.path.isdir(os.path.join(fg_parent, item)):
                        self.fg_sources.append({
                            'type': 'video',
                            'fg_path': os.path.join(fg_parent, item),
                            'alpha_path': os.path.join(alpha_parent, item)
                        })
            elif source_type == 'image':
                for fg_filename in os.listdir(fg_parent):
                    base_name, _ = os.path.splitext(fg_filename)
                    alpha_path = os.path.join(alpha_parent, base_name + '.png')
                    if os.path.isfile(alpha_path):
                        self.fg_sources.append({
                            'type': 'image',
                            'fg_path': os.path.join(fg_parent, fg_filename),
                            'alpha_path': alpha_path
                        })

        self.bg_video_paths = [os.path.join(p, d) for p in config.get('background_sources', []) for d in os.listdir(p)]

        source_counts = Counter(s['type'] for s in self.fg_sources)
        print("--- 📊 Dataloader Source Summary ---")
        print(
            f"✅ Found {len(self.fg_sources)} total foregrounds ({source_counts['image']} images, {source_counts['video']} videos).")
        print(f"✅ Found {len(self.bg_video_paths)} background sources.")
        print("----------------------------------")

    def __len__(self):
        """Returns the number of videos to generate in one epoch."""
        return self.epoch_size

    def _get_augmented_image(self, fg_img, alpha_img, frame_idx, params):
        """Applies geometric augmentations to a static foreground image."""
        current_angle = params['start_angle'] + (params['angle_speed'] * frame_idx)
        current_scale = params['zoom_center'] + params['zoom_amplitude'] * np.sin(
            (2 * np.pi * frame_idx) / params['zoom_cycle_frames'])
        dx = int(params['max_dx'] * np.sin(
            (2 * np.pi * frame_idx) / params['tx_cycle_frames'] + params['tx_start_phase']))
        dy = int(params['max_dy'] * np.sin(
            (2 * np.pi * frame_idx) / params['ty_cycle_frames'] + params['ty_start_phase']))

        scaled_size = (int(fg_img.width * current_scale), int(fg_img.height * current_scale))
        scaled_fg = fg_img.resize(scaled_size, resample=Image.BICUBIC)
        scaled_alpha = alpha_img.resize(scaled_size, resample=Image.BILINEAR)

        canvas_fg, canvas_alpha = Image.new('RGB', fg_img.size), Image.new('L', alpha_img.size)
        paste_pos = (((fg_img.width - scaled_fg.width) // 2) + dx, ((fg_img.height - scaled_fg.height) // 2) + dy)
        canvas_fg.paste(scaled_fg, paste_pos)
        canvas_alpha.paste(scaled_alpha, paste_pos)

        final_fg = canvas_fg.rotate(current_angle, resample=Image.BICUBIC)
        final_alpha = canvas_alpha.rotate(current_angle, resample=Image.BILINEAR)
        return final_fg, final_alpha

    def __getitem__(self, idx):
        """
        Generates a single synthetic video clip and its corresponding masks.
        This method contains the core logic from `generate_video`.
        """
        # Use idx for reproducible randomness
        random_state = np.random.RandomState(idx)

        target_w, target_h = self.output_resolution

        # --- 1. Select Foreground and Background Assets ---
        num_fg_instances = random_state.randint(self.min_instances_arg, self.max_instances_arg + 1)
        selected_fg_indices = random_state.choice(len(self.fg_sources), num_fg_instances, replace=False)
        selected_sources = [self.fg_sources[i] for i in selected_fg_indices]
        selected_bg_folder = random.choice(self.bg_video_paths)

        bg_frame_paths = get_files_from_folder(selected_bg_folder)
        if not bg_frame_paths: return self.__getitem__((idx + 1) % self.epoch_size)  # Retry

        # Determine the number of frames to generate
        min_clip_length = len(bg_frame_paths)
        for source in selected_sources:
            if source['type'] == 'video':
                min_clip_length = min(min_clip_length, len(get_files_from_folder(source['fg_path'])))
        num_frames_to_generate = min(min_clip_length, self.num_frames)

        if num_frames_to_generate <= 1: return self.__getitem__((idx + 1) % self.epoch_size)  # Retry

        # --- 2. Set up initial properties for each instance ---
        instance_properties = {}
        for i, source in enumerate(selected_sources):
            try:
                frame_paths = get_files_from_folder(source['fg_path'])
                with Image.open(frame_paths[0] if source['type'] == 'video' else source['fg_path']) as temp_img:
                    w, h = temp_img.size
            except:
                continue  # Skip if image is invalid

            scale = random_state.uniform(self.fg_scale_range_arg[0], self.fg_scale_range_arg[1])
            new_w = int(target_w * scale) if w > h else int(int(target_h * scale) * w / h)
            new_h = int(new_w * h / w) if w > h else int(target_h * scale)

            pos_x = random_state.randint(0, target_w - new_w) if target_w > new_w else 0
            pos_y = random_state.randint(0, target_h - new_h) if target_h > new_h else 0
            instance_properties[i] = {'size': (new_w, new_h), 'position': (pos_x, pos_y)}

        # --- 3. Pre-load images and augmentation params for static FGs ---
        loaded_images = {}
        augmentation_params = {}
        for i, source in enumerate(selected_sources):
            if source['type'] == 'image':
                fg_img = Image.open(source['fg_path']).convert("RGB")
                alpha_img = Image.open(source['alpha_path']).convert('L')
                loaded_images[i] = {'fg': fg_img, 'alpha': alpha_img}

                # Set up random motion parameters
                angle_speed = random_state.uniform(-self.rotation_speed_arg, self.rotation_speed_arg)
                zoom_min, zoom_max = random_state.uniform(self.zoom_range_arg[0], 1.0), random_state.uniform(1.0,
                                                                                                             self.zoom_range_arg[
                                                                                                                 1])
                translation_amp = random_state.uniform(self.translation_range_arg[0], self.translation_range_arg[1])

                augmentation_params[i] = {
                    'angle_speed': angle_speed, 'start_angle': random_state.uniform(-25, 25),
                    'zoom_center': (zoom_max + zoom_min) / 2, 'zoom_amplitude': (zoom_max - zoom_min) / 2,
                    'zoom_cycle_frames': random_state.randint(num_frames_to_generate, num_frames_to_generate * 2),
                    'max_dx': int(target_w * translation_amp), 'max_dy': int(target_h * translation_amp),
                    'tx_cycle_frames': random_state.randint(num_frames_to_generate, num_frames_to_generate * 2),
                    'ty_cycle_frames': random_state.randint(num_frames_to_generate, num_frames_to_generate * 2),
                    'tx_start_phase': random_state.uniform(0, 2 * np.pi),
                    'ty_start_phase': random_state.uniform(0, 2 * np.pi),
                }

        # --- 4. Generate frames one by one in memory ---
        composite_pils = []
        visible_alpha_pils = [[] for _ in range(num_fg_instances)]

        for frame_idx in range(num_frames_to_generate):
            bg_canvas = Image.open(bg_frame_paths[frame_idx]).convert("RGB").resize(self.output_resolution,
                                                                                    Image.LANCZOS)

            # Load and position all instance layers for the current frame
            instance_layers = []
            for i, source in enumerate(selected_sources):
                props = instance_properties.get(i)
                if not props: continue

                fg_img, alpha_img = None, None
                if source['type'] == 'video':
                    fg_img = Image.open(get_files_from_folder(source['fg_path'])[frame_idx]).convert("RGB")
                    alpha_img = Image.open(get_files_from_folder(source['alpha_path'])[frame_idx]).convert("L")
                elif source['type'] == 'image':
                    fg_img, alpha_img = self._get_augmented_image(
                        loaded_images[i]['fg'], loaded_images[i]['alpha'], frame_idx, augmentation_params[i]
                    )

                if fg_img and alpha_img:
                    instance_layers.append({
                        'fg': fg_img.resize(props['size'], Image.LANCZOS),
                        'alpha': alpha_img.resize(props['size'], Image.BILINEAR),
                        'position': props['position']
                    })

            # Composite layers and calculate visible alphas
            final_composite = bg_canvas.copy()
            for layer in instance_layers:
                final_composite.paste(layer['fg'], layer['position'], layer['alpha'])

            full_frame_alphas_np = []
            for layer in instance_layers:
                full_alpha = Image.new('L', self.output_resolution)
                full_alpha.paste(layer['alpha'], layer['position'])
                full_frame_alphas_np.append(np.array(full_alpha))

            for i in range(len(instance_layers)):
                visible_alpha_np = full_frame_alphas_np[i].astype(np.float32) / 255.0
                for j in range(i + 1, len(full_frame_alphas_np)):
                    visible_alpha_np *= (1.0 - (full_frame_alphas_np[j].astype(np.float32) / 255.0))
                visible_alpha_pils[i].append(Image.fromarray((visible_alpha_np * 255).astype(np.uint8)))

            composite_pils.append(final_composite)

        # --- 5. Choose one instance to be the target for this sample ---
        target_instance_idx = random_state.randint(0, num_fg_instances)
        final_alpha_pils = visible_alpha_pils[target_instance_idx]

        # --- 6. Post-process and augment the chosen mask ---
        binary_mask_pils = [
            alpha.point(lambda p: 255 if p > self.binary_mask_threshold else 0, mode='L')
            for alpha in final_alpha_pils
        ]

        if self.mask_augmentation != 'none':
            chosen_downsample_factor = random.choice(self.downsample_factors)

            if self.mask_augmentation == 'polygon':
                binary_mask_pils = [augment_to_polygon_with_nested_holes(frame, self.simplification_tolerance) for frame
                                    in binary_mask_pils]
            elif self.mask_augmentation == 'downsample':
                binary_mask_pils = [augment_by_resizing(frame, chosen_downsample_factor) for frame in binary_mask_pils]
            elif self.mask_augmentation == 'bounding_box':
                binary_mask_pils = [augment_to_bounding_box(frame) for frame in binary_mask_pils]
            elif self.mask_augmentation == 'instability':
                binary_mask_pils = [augment_with_instability(frame, self.instability_noise_level) for frame in
                                    binary_mask_pils]
            elif self.mask_augmentation == 'all':
                binary_mask_pils = [
                    apply_all_augmentations(frame, chosen_downsample_factor, self.simplification_tolerance,
                                            self.instability_noise_level)
                    for frame in binary_mask_pils]

        if random.random() < self.temporal_augmentation_rate:
            binary_mask_pils = augment_with_temporal_occlusion(
                binary_mask_pils, self.num_occlusions, self.occlusion_shape, self.occlusion_scale_range
            )

        # --- 7. Convert PIL lists to Tensors ---
        video_tensor = torch.stack([self.to_tensor(img) for img in composite_pils])
        alpha_tensor = torch.stack([self.to_tensor(img) for img in final_alpha_pils])
        mask_tensor = torch.stack([self.to_tensor(img) for img in binary_mask_pils])

        # Normalize to -1 to 1 range
        video_tensor = video_tensor * 2.0 - 1.0
        alpha_tensor = alpha_tensor * 2.0 - 1.0
        mask_tensor = mask_tensor * 2.0 - 1.0

        return {
            'video': video_tensor,
            'alpha': alpha_tensor,
            'binary_mask': mask_tensor
        }


if __name__ == '__main__':
    # ============================================================================
    # Example Usage
    # ============================================================================

    # 1. Create a dummy config file (e.g., 'synthetic_config.yaml')
    #    You MUST replace the paths below with paths to your actual assets.
    #    - fg_path should contain subfolders of video frames or individual images.
    #    - alpha_path should mirror the structure of fg_path with corresponding masks.
    #    - background_sources should contain folders of video frame sequences.

    config_content = """
foreground_sources:
  - type: 'image'
    fg_path: './path/to/your/foreground_images'
    alpha_path: './path/to/your/foreground_alphas'
  - type: 'video'
    fg_path: './path/to/your/foreground_videos'
    alpha_path: './path/to/your/foreground_video_alphas'

background_sources:
  - './path/to/your/backgrounds'
"""
    with open('synthetic_config.yaml', 'w') as f:
        f.write(config_content)

    print("Created dummy 'synthetic_config.yaml'. Please edit it with your actual asset paths.")

    # 2. Instantiate the dataloader
    #    This will fail if the paths in the config are not valid.
    try:
        dataset = OnTheFlySyntheticDataset(
            config_file='synthetic_config.yaml',
            epoch_size=100,
            num_frames=8,
            height=256,
            width=256,
            mask_augmentation='polygon'
        )

        # 3. Get a sample
        sample_data = dataset[0]  # Get the first generated sample

        # 4. Print information about the sample
        print(f"\nSuccessfully generated a sample!")
        for key, tensor in sample_data.items():
            print(f"- Output '{key}' tensor shape: {tensor.shape}")  # Shape: (Frames, Channels, Height, Width)

        # Example of how to save the first frame of the output for verification
        from torchvision.utils import save_image

        # Denormalize from [-1, 1] to [0, 1] before saving
        video_frame_0 = (sample_data['video'][0] + 1) / 2
        alpha_frame_0 = (sample_data['alpha'][0] + 1) / 2
        mask_frame_0 = (sample_data['binary_mask'][0] + 1) / 2

        save_image(video_frame_0, 'sample_video_frame.jpg')
        save_image(alpha_frame_0, 'sample_alpha_frame.png')
        save_image(mask_frame_0, 'sample_mask_frame.png')
        print("\nSaved the first frame of a sample to 'sample_video_frame.jpg' etc. for you to inspect.")

    except Exception as e:
        print(f"\nCould not run example. Please ensure 'synthetic_config.yaml' contains valid paths.")
        print(f"Error: {e}")

