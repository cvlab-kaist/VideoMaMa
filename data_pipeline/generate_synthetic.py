import os
import cv2
import numpy as np
import argparse
from PIL import Image, ImageOps
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import random as py_random
import yaml
from functools import partial
import tarfile
import shutil
from collections import Counter  # <-- ADDED IMPORT


def get_files_from_folder(folder_path):
    """Helper function to get sorted image files from a directory."""
    if not os.path.isdir(folder_path): return []
    files = []
    for f_bytes in os.listdir(folder_path):
        try:
            f_str = f_bytes.decode('utf-8')
        except (UnicodeDecodeError, AttributeError):
            f_str = f_bytes
        if isinstance(f_str, str) and f_str.lower().endswith(('.png', '.jpg', '.jpeg')):
            files.append(os.path.join(folder_path, f_str))
    return sorted(files)


def generate_video(video_id, fg_sources, video_indices, image_indices, bg_video_paths, output_dir, image_output_dir,
                   alpha_output_dir, bg_output_dir,
                   num_frames_arg, min_instances_arg, max_instances_arg, include_bgs_arg,
                   output_resolution, fg_scale_range_arg, instance_ratio_weights, rotation_speed_arg,
                   translation_range_arg,
                   zoom_range_arg, save_mode_arg):
    """
    Generates a video by layering foregrounds over a background.
    """
    random = np.random.RandomState(video_id)
    if not fg_sources or not bg_video_paths: return []

    target_w, target_h = output_resolution
    video_weight, image_weight = instance_ratio_weights
    source_type_population = ['video'] * video_weight + ['image'] * image_weight

    MAX_SETUP_ATTEMPTS = 10
    for setup_attempt in range(MAX_SETUP_ATTEMPTS):
        max_allowed_instances = min(max_instances_arg, len(fg_sources))
        if min_instances_arg > max_allowed_instances: return []
        num_fg_instances = random.randint(min_instances_arg, max_allowed_instances + 1)

        selected_fg_indices = []
        temp_video_indices = video_indices[:]
        random.shuffle(temp_video_indices)
        temp_image_indices = image_indices[:]
        random.shuffle(temp_image_indices)

        for _ in range(num_fg_instances):
            choice = random.choice(source_type_population)
            if choice == 'video' and temp_video_indices:
                selected_fg_indices.append(temp_video_indices.pop())
            elif temp_image_indices:
                selected_fg_indices.append(temp_image_indices.pop())
            elif temp_video_indices:
                selected_fg_indices.append(temp_video_indices.pop())
            else:
                break
        if not selected_fg_indices: continue

        selected_bg_folder = random.choice(bg_video_paths)
        selected_sources = [fg_sources[i] for i in selected_fg_indices]

        bg_frame_paths = get_files_from_folder(selected_bg_folder)
        if not bg_frame_paths: continue
        min_clip_length = len(bg_frame_paths)
        for source in selected_sources:
            if source['type'] == 'video':
                min_clip_length = min(min_clip_length, len(get_files_from_folder(source['fg_path'])))
        num_frames_to_generate = min(min_clip_length, num_frames_arg)
        if num_frames_to_generate <= 0: continue

        instance_properties = {}
        for i, source_idx in enumerate(selected_fg_indices):
            source = fg_sources[source_idx]
            try:
                if source['type'] == 'image':
                    with Image.open(source['fg_path']) as temp_img:
                        w, h = temp_img.size
                elif source['type'] == 'video':
                    frame_paths = get_files_from_folder(source['fg_path'])
                    if not frame_paths: raise FileNotFoundError
                    with Image.open(frame_paths[0]) as temp_img:
                        w, h = temp_img.size
                else:
                    continue
            except FileNotFoundError:
                continue

            scale = random.uniform(fg_scale_range_arg[0], fg_scale_range_arg[1])
            new_w = int(target_w * scale) if w > h else int(int(target_h * scale) * w / h)
            new_h = int(new_w * h / w) if w > h else int(target_h * scale)

            pos_x = random.randint(0, target_w - new_w) if target_w > new_w else 0
            pos_y = random.randint(0, target_h - new_h) if target_h > new_h else 0
            instance_properties[i] = {'size': (new_w, new_h), 'position': (pos_x, pos_y)}

        loaded_images = {}
        augmentation_params = {}
        for i, source_idx in enumerate(selected_fg_indices):
            source = fg_sources[source_idx]
            if source['type'] == 'image':
                fg_img = Image.open(source['fg_path']).convert("RGB")
                alpha_img = Image.open(source['alpha_path']).convert('L')
                loaded_images[i] = {'fg': fg_img, 'alpha': alpha_img}

                angle_speed = random.uniform(rotation_speed_arg * 0.2, rotation_speed_arg) * random.choice([-1, 1])
                start_angle = random.uniform(-25, 25) if random.random() > 0.5 else 0

                zoom_min, zoom_max = random.uniform(zoom_range_arg[0], 1.0), random.uniform(1.0, zoom_range_arg[1])
                zoom_center, zoom_amplitude = (zoom_max + zoom_min) / 2, (zoom_max - zoom_min) / 2
                zoom_cycle_frames = random.randint(max(num_frames_to_generate, 60), max(num_frames_to_generate, 60) * 2)

                translation_amplitude = random.uniform(translation_range_arg[0], translation_range_arg[1])
                max_dx, max_dy = int(target_w * translation_amplitude), int(target_h * translation_amplitude)
                tx_cycle_frames = random.randint(max(num_frames_to_generate, 60), max(num_frames_to_generate, 60) * 2)
                ty_cycle_frames = random.randint(max(num_frames_to_generate, 60), max(num_frames_to_generate, 60) * 2)
                tx_start_phase, ty_start_phase = random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi)

                augmentation_params[i] = {
                    'angle_speed': angle_speed, 'start_angle': start_angle, 'zoom_center': zoom_center,
                    'zoom_amplitude': zoom_amplitude, 'zoom_cycle_frames': zoom_cycle_frames,
                    'max_dx': max_dx, 'max_dy': max_dy, 'tx_cycle_frames': tx_cycle_frames,
                    'ty_cycle_frames': ty_cycle_frames, 'tx_start_phase': tx_start_phase,
                    'ty_start_phase': ty_start_phase,
                }

        def get_augmented_image(instance_idx, frame_idx):
            params = augmentation_params[instance_idx]
            base_fg, base_alpha = loaded_images[instance_idx]['fg'], loaded_images[instance_idx]['alpha']

            current_angle = params['start_angle'] + (params['angle_speed'] * frame_idx)
            current_scale = params['zoom_center'] + params['zoom_amplitude'] * np.sin(
                (2 * np.pi * frame_idx) / params['zoom_cycle_frames'])
            dx = int(params['max_dx'] * np.sin(
                (2 * np.pi * frame_idx) / params['tx_cycle_frames'] + params['tx_start_phase']))
            dy = int(params['max_dy'] * np.sin(
                (2 * np.pi * frame_idx) / params['ty_cycle_frames'] + params['ty_start_phase']))

            scaled_size = (int(base_fg.width * current_scale), int(base_fg.height * current_scale))
            scaled_fg = base_fg.resize(scaled_size, resample=Image.BICUBIC)
            scaled_alpha = base_alpha.resize(scaled_size, resample=Image.BILINEAR)

            canvas_fg, canvas_alpha = Image.new('RGB', base_fg.size), Image.new('L', base_alpha.size)
            paste_pos = (((base_fg.width - scaled_fg.width) // 2) + dx, ((base_fg.height - scaled_fg.height) // 2) + dy)
            canvas_fg.paste(scaled_fg, paste_pos)
            canvas_alpha.paste(scaled_alpha, paste_pos)

            final_fg = canvas_fg.rotate(current_angle, resample=Image.BICUBIC)
            final_alpha = canvas_alpha.rotate(current_angle, resample=Image.BILINEAR)
            return final_fg, final_alpha

        first_frame_full_alphas = []
        for i, source_idx in enumerate(selected_fg_indices):
            source, props = fg_sources[source_idx], instance_properties[i]
            if source['type'] == 'video':
                alpha_img = Image.open(get_files_from_folder(source['alpha_path'])[0]).convert("L")
            elif source['type'] == 'image':
                _, alpha_img = get_augmented_image(i, 0)
            if alpha_img:
                full_alpha = Image.new('L', (target_w, target_h))
                full_alpha.paste(alpha_img.resize(props['size'], resample=Image.BILINEAR), props['position'])
                first_frame_full_alphas.append(full_alpha)

        all_alpha_layers_np = [np.array(alpha) for alpha in first_frame_full_alphas]
        is_composition_valid = True
        for i in range(len(all_alpha_layers_np)):
            visible_alpha_np = all_alpha_layers_np[i].astype(np.float32) / 255.0
            for j in range(i + 1, len(all_alpha_layers_np)):
                visible_alpha_np *= (1.0 - (all_alpha_layers_np[j].astype(np.float32) / 255.0))
            if np.sum(visible_alpha_np) < 50:
                is_composition_valid = False;
                break
        if is_composition_valid: break
    else:
        return []

    padded_video_id = f'{video_id:06d}'
    video_image_dir = os.path.join(image_output_dir, padded_video_id)
    video_alpha_dir = os.path.join(alpha_output_dir, padded_video_id)
    video_bg_dir = os.path.join(bg_output_dir, padded_video_id)
    os.makedirs(video_image_dir, exist_ok=True)
    os.makedirs(video_alpha_dir, exist_ok=True)
    if include_bgs_arg: os.makedirs(video_bg_dir, exist_ok=True)

    for frame_idx in range(num_frames_to_generate):
        bg_canvas = Image.open(bg_frame_paths[frame_idx]).convert("RGB").resize((target_w, target_h),
                                                                                resample=Image.LANCZOS)
        instance_layers_for_frame = []
        for i, source_idx in enumerate(selected_fg_indices):
            source, props = fg_sources[source_idx], instance_properties[i]
            fg_img, alpha_img = None, None
            if source['type'] == 'video':
                fg_img = Image.open(get_files_from_folder(source['fg_path'])[frame_idx]).convert("RGB")
                alpha_img = Image.open(get_files_from_folder(source['alpha_path'])[frame_idx]).convert("L")
            elif source['type'] == 'image':
                fg_img, alpha_img = get_augmented_image(i, frame_idx)
            if fg_img and alpha_img:
                instance_layers_for_frame.append({
                    'fg': fg_img.resize(props['size'], resample=Image.LANCZOS),
                    'alpha': alpha_img.resize(props['size'], resample=Image.BILINEAR),
                    'position': props['position']
                })

        final_composite_image = bg_canvas.copy()
        for layer in instance_layers_for_frame:
            final_composite_image.paste(layer['fg'], layer['position'], layer['alpha'])

        full_frame_alphas = []
        for layer in instance_layers_for_frame:
            full_alpha = Image.new('L', (target_w, target_h))
            full_alpha.paste(layer['alpha'], layer['position'])
            full_frame_alphas.append(np.array(full_alpha))

        visible_alphas = []
        for i in range(len(instance_layers_for_frame)):
            visible_alpha_np = full_frame_alphas[i].astype(np.float32) / 255.0
            for j in range(i + 1, len(full_frame_alphas)):
                visible_alpha_np *= (1.0 - (full_frame_alphas[j].astype(np.float32) / 255.0))
            visible_alphas.append(Image.fromarray((visible_alpha_np * 255).astype(np.uint8)))

        final_composite_image.save(os.path.join(video_image_dir, f'{frame_idx:04d}.jpg'))
        for i, visible_alpha_pil in enumerate(visible_alphas):
            instance_alpha_dir = os.path.join(video_alpha_dir, str(i));
            os.makedirs(instance_alpha_dir, exist_ok=True)
            visible_alpha_pil.save(os.path.join(instance_alpha_dir, f'{frame_idx:04d}.png'))
            if include_bgs_arg:
                unique_bg = bg_canvas.copy()
                for j, other_layer in enumerate(instance_layers_for_frame):
                    if i != j: unique_bg.paste(other_layer.get('fg'), other_layer.get('position'),
                                               other_layer.get('alpha'))
                instance_bg_dir = os.path.join(video_bg_dir, str(i));
                os.makedirs(instance_bg_dir, exist_ok=True)
                unique_bg.save(os.path.join(instance_bg_dir, f'{frame_idx:04d}.jpg'))

    parquet_records = []
    if save_mode_arg == 'tar':
        archive_rel_path = os.path.join('archives', f'{padded_video_id}.tar')
        archive_full_path = os.path.join(output_dir, archive_rel_path)
        os.makedirs(os.path.dirname(archive_full_path), exist_ok=True)
        with tarfile.open(archive_full_path, "w") as tar:
            tar.add(video_image_dir, arcname=os.path.basename(video_image_dir))
            tar.add(video_alpha_dir, arcname=os.path.basename(video_alpha_dir))
            if include_bgs_arg: tar.add(video_bg_dir, arcname=os.path.basename(video_bg_dir))
        shutil.rmtree(video_image_dir);
        shutil.rmtree(video_alpha_dir)
        if include_bgs_arg: shutil.rmtree(video_bg_dir)

        for i, source_idx in enumerate(selected_fg_indices):
            record = {
                'video_id': padded_video_id, 'instance_id': i, 'type': fg_sources[source_idx]['type'],
                'archive_path': archive_rel_path,
                'composite_path_template': os.path.join(padded_video_id, '{frame:04d}.jpg'),
                'alpha_path_template': os.path.join(padded_video_id, str(i), '{frame:04d}.png'),
                'bg_path_template': os.path.join(padded_video_id, str(i),
                                                 '{frame:04d}.jpg') if include_bgs_arg else None,
                'num_frames': num_frames_to_generate,
            }
            parquet_records.append(record)
    else:
        for i, source_idx in enumerate(selected_fg_indices):
            record = {
                'video_id': padded_video_id, 'instance_id': i, 'type': fg_sources[source_idx]['type'],
                'composite_path': os.path.relpath(video_image_dir, output_dir),
                'alpha_path': os.path.relpath(os.path.join(video_alpha_dir, str(i)), output_dir),
                'bg_path': os.path.relpath(os.path.join(video_bg_dir, str(i)), output_dir) if include_bgs_arg else None,
                'num_frames': num_frames_to_generate,
            }
            parquet_records.append(record)
    return parquet_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic video data.")
    parser.add_argument('--config-file', type=str, required=True, help='Path to the YAML config file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save generated data.')
    parser.add_argument('--resolution', type=str, default='1024x1024', help='Output resolution in WxH format.')
    parser.add_argument('--instance-ratio', type=str, default='1:1', help='Ratio of video to image instances.')
    parser.add_argument('--max-num-videos', type=int, default=100, help='Total number of videos to generate.')
    parser.add_argument('--num-frames', type=int, default=150, help='Maximum frames per video.')
    parser.add_argument('--min-instances', type=int, default=1, help='Minimum foreground instances.')
    parser.add_argument('--max-instances', type=int, default=3, help='Maximum foreground instances.')
    parser.add_argument('--include-bgs', action='store_true', help='Generate unique backgrounds for each instance.')
    parser.add_argument('--n-workers', type=int, default=8, help='Number of worker processes.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--fg-scale-range', type=str, default='0.5,0.99', help='Min/max FG scale.')
    parser.add_argument('--rotation-speed', type=float, default=1.25, help='Max rotation speed in degrees/frame.')
    parser.add_argument('--zoom-range', type=str, default='0.8,1.2', help='Min/max zoom scale.')
    parser.add_argument('--translation-range', type=str, default='0.3,0.5', help='Min/max translation amplitude.')
    parser.add_argument('--save-mode', type=str, default='tar', choices=['tar', 'files'],
                        help='Save method: "tar" or "files".')
    args = parser.parse_args()

    np.random.seed(args.seed);
    py_random.seed(args.seed)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    output_resolution = tuple(map(int, args.resolution.split('x')))
    fg_scale_range = tuple(map(float, args.fg_scale_range.split(',')))
    zoom_range = tuple(map(float, args.zoom_range.split(',')))
    video_weight, image_weight = map(int, args.instance_ratio.split(':'))
    translation_range = tuple(map(float, args.translation_range.split(',')))

    fg_sources = []
    for source_def in config.get('foreground_sources', []):
        source_type = source_def.get('type')
        fg_parent = source_def.get('fg_path')
        alpha_parent = source_def.get('alpha_path')

        if not all([source_type, fg_parent, alpha_parent]):
            print(f"Skipping incomplete source definition: {source_def}")
            continue

        if source_type == 'video':
            for item in os.listdir(fg_parent):
                if os.path.isdir(os.path.join(fg_parent, item)) and os.path.isdir(os.path.join(alpha_parent, item)):
                    fg_sources.append({
                        'type': 'video',
                        'fg_path': os.path.join(fg_parent, item),
                        'alpha_path': os.path.join(alpha_parent, item)
                    })
        elif source_type == 'image':
            possible_alpha_exts = ['.png', '.jpg', '.jpeg']
            for fg_filename in os.listdir(fg_parent):
                full_fg_path = os.path.join(fg_parent, fg_filename)
                if not os.path.isfile(full_fg_path):
                    continue
                base_name, _ = os.path.splitext(fg_filename)
                for alpha_ext in possible_alpha_exts:
                    potential_alpha_name = base_name + alpha_ext
                    full_alpha_path = os.path.join(alpha_parent, potential_alpha_name)
                    if os.path.isfile(full_alpha_path):
                        fg_sources.append({
                            'type': 'image',
                            'fg_path': full_fg_path,
                            'alpha_path': full_alpha_path
                        })
                        break

    bg_video_paths = [os.path.join(p, d) for p in config.get('background_sources', []) if os.path.isdir(p) for d in
                      os.listdir(p) if os.path.isdir(os.path.join(p, d))]

    # --- START: Added code for annotation summary ---
    print("\n--- 📊 Annotation Summary ---")
    if not fg_sources:
        print("⚠️  No foreground annotations found. Please check your config file and paths.")
    else:
        print(f"✅ Found {len(fg_sources)} total foreground annotations.")
        source_counts = Counter(s['type'] for s in fg_sources)
        for source_type, count in source_counts.items():
            print(f"  - {source_type.capitalize()}s: {count}")

    print(f"✅ Found {len(bg_video_paths)} background video sources.")
    if not bg_video_paths:
        print("⚠️  No background sources found. The script might fail if backgrounds are required.")
    print("--------------------------\n")
    # --- END: Added code for annotation summary ---

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    image_output_dir, alpha_output_dir, bg_output_dir = [os.path.join(output_dir, d) for d in
                                                         ['images', 'alphas', 'bg']]

    worker_func = partial(generate_video,
                          fg_sources=fg_sources,
                          video_indices=[i for i, s in enumerate(fg_sources) if s['type'] == 'video'],
                          image_indices=[i for i, s in enumerate(fg_sources) if s['type'] == 'image'],
                          bg_video_paths=bg_video_paths, output_dir=output_dir, image_output_dir=image_output_dir,
                          alpha_output_dir=alpha_output_dir, bg_output_dir=bg_output_dir,
                          num_frames_arg=args.num_frames, min_instances_arg=args.min_instances,
                          max_instances_arg=args.max_instances, include_bgs_arg=args.include_bgs,
                          output_resolution=output_resolution, fg_scale_range_arg=fg_scale_range,
                          instance_ratio_weights=(video_weight, image_weight), rotation_speed_arg=args.rotation_speed,
                          translation_range_arg=translation_range, zoom_range_arg=zoom_range,
                          save_mode_arg=args.save_mode)

    all_records = []
    with Pool(args.n_workers) as p:
        pbar = tqdm(p.imap(worker_func, range(args.max_num_videos)), total=args.max_num_videos,
                    desc="Generating Videos")
        for records in pbar:
            if records: all_records.extend(records)

    if all_records:
        df = pd.DataFrame(all_records).sort_values(by=['video_id', 'instance_id'])
        df.to_parquet(os.path.join(output_dir, 'metadata.parquet'), index=False)
        print(f"\n🚀 Generation complete. Saved metadata for {len(df)} instances.")
    else:
        print("\nNo valid videos or annotations were generated.")