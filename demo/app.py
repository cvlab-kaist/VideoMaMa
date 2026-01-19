"""
VideoMaMa Gradio Demo
Interactive video matting with SAM2 mask tracking
"""

import sys
sys.path.append("../")
sys.path.append("../../")

import os
import json
import time
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path

from sam2_wrapper import load_sam2_tracker
from videomama_wrapper import load_videomama_pipeline, videomama
from tools.painter import mask_painter, point_painter

import warnings
warnings.filterwarnings("ignore")

# Global models
sam2_tracker = None
videomama_pipeline = None

# Constants
MASK_COLOR = 3
MASK_ALPHA = 0.7
CONTOUR_COLOR = 1
CONTOUR_WIDTH = 5
POINT_COLOR_POS = 8   # Positive points - orange
POINT_COLOR_NEG = 1   # Negative points - red
POINT_ALPHA = 0.9
POINT_RADIUS = 15

def initialize_models():
    """Initialize SAM2 and VideoMaMa models"""
    global sam2_tracker, videomama_pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load SAM2
    sam2_tracker = load_sam2_tracker(device=device)
    
    # Load VideoMaMa
    videomama_pipeline = load_videomama_pipeline(device=device)
    
    print("All models initialized successfully!")


def extract_frames_from_video(video_path, max_frames=24):
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (default: 24)
        
    Returns:
        frames: List of numpy arrays (H,W,3), uint8 RGB
        adjusted_fps: Adjusted FPS for output video to maintain normal playback speed
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read all frames first
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
    
    cap.release()
    
    # If video has more frames than max_frames, randomly sample
    if len(all_frames) > max_frames:
        print(f"Video has {len(all_frames)} frames, randomly sampling {max_frames} frames...")
        # Sort indices to maintain temporal order
        sampled_indices = sorted(np.random.choice(len(all_frames), max_frames, replace=False))
        frames = [all_frames[i] for i in sampled_indices]
        print(f"Sampled frame indices: {sampled_indices}")
        
        # Adjust FPS to maintain normal playback speed
        # If we sampled N frames from M total frames, adjust FPS proportionally
        adjusted_fps = original_fps * (len(frames) / len(all_frames))
    else:
        frames = all_frames
        adjusted_fps = original_fps
        print(f"Video has {len(frames)} frames (≤ {max_frames}), using all frames")
    
    print(f"Using {len(frames)} frames from video (Original FPS: {original_fps:.2f}, Adjusted FPS: {adjusted_fps:.2f})")
    
    return frames, adjusted_fps


def get_prompt(click_state, click_input):
    """
    Convert click input to prompt format
    
    Args:
        click_state: [[points], [labels]]
        click_input: JSON string "[[x, y, label]]"
        
    Returns:
        Updated click_state
    """
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    
    for input_item in inputs:
        points.append(input_item[:2])
        labels.append(input_item[2])
    
    click_state[0] = points
    click_state[1] = labels
    
    return click_state


def load_video(video_input, video_state, num_frames):
    """
    Load video and extract first frame for mask generation
    """
    # Clean up old output files if they exist
    if video_state is not None and "output_paths" in video_state:
        cleanup_old_videos(video_state["output_paths"])
    
    if video_input is None:
        return video_state, None, \
               gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=False), gr.update(visible=False)
    
    # Extract frames with user-specified number
    frames, fps = extract_frames_from_video(video_input, max_frames=num_frames)
    
    if len(frames) == 0:
        return video_state, None, \
               gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=False), gr.update(visible=False)
    
    # Initialize video state
    video_state = {
        "frames": frames,
        "fps": fps,
        "first_frame_mask": None,
        "masks": None,
    }
    
    first_frame_pil = Image.fromarray(frames[0])
    
    return video_state, first_frame_pil, \
           gr.update(visible=True), gr.update(visible=True), \
           gr.update(visible=True), gr.update(visible=False)


def sam_refine(video_state, point_prompt, click_state, evt: gr.SelectData):
    """
    Add click and update mask on first frame
    
    Args:
        video_state: Dictionary with video data
        point_prompt: "Positive" or "Negative"
        click_state: [[points], [labels]]
        evt: Gradio SelectData event with click coordinates
    """
    if video_state is None or "frames" not in video_state:
        return None, video_state, click_state
    
    # Add new click
    x, y = evt.index[0], evt.index[1]
    label = 1 if point_prompt == "Positive" else 0
    
    click_state[0].append([x, y])
    click_state[1].append(label)
    
    print(f"Added {point_prompt} click at ({x}, {y}). Total clicks: {len(click_state[0])}")
    
    # Generate mask with SAM2
    first_frame = video_state["frames"][0]
    mask = sam2_tracker.get_first_frame_mask(
        frame=first_frame,
        points=click_state[0],
        labels=click_state[1]
    )
    
    # Store mask in video state
    video_state["first_frame_mask"] = mask
    
    # Visualize mask and points
    painted_image = mask_painter(
        first_frame.copy(),
        mask,
        MASK_COLOR,
        MASK_ALPHA,
        CONTOUR_COLOR,
        CONTOUR_WIDTH
    )
    
    # Paint positive points
    positive_points = np.array([click_state[0][i] for i in range(len(click_state[0])) 
                               if click_state[1][i] == 1])
    if len(positive_points) > 0:
        painted_image = point_painter(
            painted_image,
            positive_points,
            POINT_COLOR_POS,
            POINT_ALPHA,
            POINT_RADIUS,
            CONTOUR_COLOR,
            CONTOUR_WIDTH
        )
    
    # Paint negative points
    negative_points = np.array([click_state[0][i] for i in range(len(click_state[0])) 
                               if click_state[1][i] == 0])
    if len(negative_points) > 0:
        painted_image = point_painter(
            painted_image,
            negative_points,
            POINT_COLOR_NEG,
            POINT_ALPHA,
            POINT_RADIUS,
            CONTOUR_COLOR,
            CONTOUR_WIDTH
        )
    
    painted_pil = Image.fromarray(painted_image)
    
    return painted_pil, video_state, click_state


def clear_clicks(video_state, click_state):
    """Clear all clicks and reset to original first frame"""
    click_state = [[], []]
    
    if video_state is not None and "frames" in video_state:
        first_frame = video_state["frames"][0]
        video_state["first_frame_mask"] = None
        return Image.fromarray(first_frame), video_state, click_state
    
    return None, video_state, click_state


def propagate_masks(video_state, click_state):
    """
    Propagate first frame mask through entire video using SAM2
    """
    if video_state is None or "frames" not in video_state:
        return video_state, "No video loaded", gr.update(visible=False)
    
    if len(click_state[0]) == 0:
        return video_state, "⚠️ Please add at least one point first", gr.update(visible=False)
    
    frames = video_state["frames"]
    
    # Track through video
    print(f"Tracking object through {len(frames)} frames...")
    masks = sam2_tracker.track_video(
        frames=frames,
        points=click_state[0],
        labels=click_state[1]
    )
    
    video_state["masks"] = masks
    
    status_msg = f"✓ Generated {len(masks)} masks. Ready to run VideoMaMa!"
    
    return video_state, status_msg, gr.update(visible=True)


def run_videomama_with_sam2(video_state, click_state):
    """
    Run SAM2 propagation and VideoMaMa inference together
    """
    if video_state is None or "frames" not in video_state:
        return video_state, None, None, None, "⚠️ No video loaded"
    
    if len(click_state[0]) == 0:
        return video_state, None, None, None, "⚠️ Please add at least one point first"
    
    frames = video_state["frames"]
    
    # Step 1: Track through video with SAM2
    print(f"🎯 Tracking object through {len(frames)} frames with SAM2...")
    masks = sam2_tracker.track_video(
        frames=frames,
        points=click_state[0],
        labels=click_state[1]
    )
    
    video_state["masks"] = masks
    print(f"✓ Generated {len(masks)} masks")
    
    # Step 2: Run VideoMaMa
    print(f"🎨 Running VideoMaMa on {len(frames)} frames...")
    output_frames = videomama(videomama_pipeline, frames, masks)
    
    # Save output videos
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    output_video_path = output_dir / f"output_{timestamp}.mp4"
    mask_video_path = output_dir / f"masks_{timestamp}.mp4"
    greenscreen_path = output_dir / f"greenscreen_{timestamp}.mp4"
    
    # Save matting result
    save_video(output_frames, output_video_path, video_state["fps"])
    
    # Save mask video (for visualization)
    mask_frames_rgb = [np.stack([m, m, m], axis=-1) for m in masks]
    save_video(mask_frames_rgb, mask_video_path, video_state["fps"])
    
    # Create greenscreen composite: RGB * VideoMaMa_alpha + green * (1 - VideoMaMa_alpha)
    # VideoMaMa output_frames already contain the alpha matte result
    greenscreen_frames = []
    for orig_frame, output_frame in zip(frames, output_frames):
        # Extract alpha matte from VideoMaMa output
        # VideoMaMa outputs matted foreground, we use its intensity as alpha
        gray = cv2.cvtColor(output_frame, cv2.COLOR_RGB2GRAY)
        alpha = np.clip(gray.astype(np.float32) / 255.0, 0, 1)
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
        
        # Create green background
        green_bg = np.zeros_like(orig_frame)
        green_bg[:, :] = [156, 251, 165]  # Green screen color
        
        # Composite: original_RGB * alpha + green * (1 - alpha)
        composite = (orig_frame.astype(np.float32) * alpha_3ch + 
                    green_bg.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)
        greenscreen_frames.append(composite)
    
    save_video(greenscreen_frames, greenscreen_path, video_state["fps"])
    
    status_msg = f"✓ Complete! Generated {len(output_frames)} frames."
    
    # Store paths for cleanup later
    video_state["output_paths"] = [str(output_video_path), str(mask_video_path), str(greenscreen_path)]
    
    return video_state, str(output_video_path), str(mask_video_path), str(greenscreen_path), status_msg


def save_video(frames, output_path, fps):
    """Save frames as video file"""
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"Saved video to {output_path}")


def cleanup_old_videos(video_paths):
    """Remove old output videos to save storage space"""
    if video_paths is None:
        return
    
    for path in video_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleaned up: {path}")
        except Exception as e:
            print(f"Failed to remove {path}: {e}")


def cleanup_old_outputs(max_age_minutes=30):
    """
    Remove output files older than max_age_minutes to prevent storage overflow
    This runs periodically to clean up abandoned files
    """
    output_dir = Path("outputs")
    if not output_dir.exists():
        return
    
    current_time = time.time()
    max_age_seconds = max_age_minutes * 60
    
    for file_path in output_dir.glob("*.mp4"):
        try:
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                print(f"Cleaned up old file: {file_path} (age: {file_age/60:.1f} minutes)")
        except Exception as e:
            print(f"Failed to clean up {file_path}: {e}")


def restart():
    """Reset all states"""
    return None, [[], []], None, \
           gr.update(visible=False), gr.update(visible=False), \
           gr.update(visible=False), None, None, None, ""


# CSS styling
custom_css = """
.gradio-container {width: 90% !important; margin: 0 auto;}
.title-text {text-align: center; font-size: 48px; font-weight: bold; 
             background: linear-gradient(to right, #8b5cf6, #10b981); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.description-text {text-align: center; font-size: 18px; margin: 20px 0;}
button {border-radius: 8px !important;}
.green_button {background-color: #10b981 !important; color: white !important;}
.red_button {background-color: #ef4444 !important; color: white !important;}
.run_matting_button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 18px !important;
    padding: 20px !important;
    box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.75) !important;
    border: none !important;
}
.run_matting_button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 50%, #f093fb 100%) !important;
    box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.9) !important;
    transform: translateY(-2px) !important;
}
"""

# Build Gradio interface
with gr.Blocks(css=custom_css, title="VideoMaMa Demo") as demo:
    gr.HTML('<div class="title-text">VideoMaMa Interactive Demo</div>')
    gr.Markdown(
        '<div class="description-text">🎬 Upload a video → 🖱️ Click to mark object → ✅ Generate masks → 🎨 Run VideoMaMa</div>'
    )
    gr.Markdown(
        '<div style="text-align: center; color: #6b7280; font-size: 14px; margin-top: -10px;">Note: VideoMaMa processes the selected number of frames (1-50). Longer videos will be randomly sampled.</div>'
    )
    
    # State variables
    video_state = gr.State(None)
    click_state = gr.State([[], []])  # [[points], [labels]]
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Upload Video")
            video_input = gr.Video(label="Input Video")
            num_frames_slider = gr.Slider(
                minimum=1,
                maximum=50,
                value=24,
                step=1,
                label="Number of Frames to Process",
                info="VideoMaMa will process only this many frames. More frames = better quality but slower."
            )
            load_button = gr.Button("📁 Load Video", variant="primary")
            
            gr.Markdown("### Step 2: Mark Object")
            point_prompt = gr.Radio(
                choices=["Positive", "Negative"],
                value="Positive",
                label="Click Type",
                info="Positive: object, Negative: background",
                visible=False
            )
            clear_button = gr.Button("🗑️ Clear Clicks", visible=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### First Frame (Click to Add Points)")
            first_frame_display = gr.Image(
                label="First Frame",
                type="pil",
                interactive=True
            )
            run_button = gr.Button("🚀 Run Matting", visible=False, elem_classes="run_matting_button", size="lg")
    
    status_text = gr.Textbox(label="Status", value="", interactive=False, visible=False)
    
    gr.Markdown("### Outputs")
    with gr.Row():
        with gr.Column():
            output_video = gr.Video(label="Matting Result", autoplay=True)
        with gr.Column():
            greenscreen_video = gr.Video(label="Greenscreen Composite", autoplay=True)
        with gr.Column():
            mask_video = gr.Video(label="Mask Track", autoplay=True)
    
    # Event handlers
    load_button.click(
        fn=load_video,
        inputs=[video_input, video_state, num_frames_slider],
        outputs=[video_state, first_frame_display, 
                point_prompt, clear_button, run_button, status_text]
    )
    
    first_frame_display.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state],
        outputs=[first_frame_display, video_state, click_state]
    )
    
    clear_button.click(
        fn=clear_clicks,
        inputs=[video_state, click_state],
        outputs=[first_frame_display, video_state, click_state]
    )
    
    run_button.click(
        fn=run_videomama_with_sam2,
        inputs=[video_state, click_state],
        outputs=[video_state, output_video, mask_video, greenscreen_video, status_text]
    )
    
    video_input.change(
        fn=restart,
        inputs=[],
        outputs=[video_state, click_state, first_frame_display,
                point_prompt, clear_button, run_button, 
                output_video, mask_video, greenscreen_video, status_text]
    )
    
    # Examples
    gr.Markdown("---\n### 📦 Example Videos")
    example_dir = Path("samples")
    if example_dir.exists():
        examples = [str(p) for p in sorted(example_dir.glob("*.mp4"))]
        if examples:
            gr.Examples(examples=examples, inputs=[video_input])


if __name__ == "__main__":
    print("=" * 60)
    print("VideoMaMa Interactive Demo")
    print("=" * 60)
    
    # Clean up old output files on startup
    cleanup_old_outputs(max_age_minutes=30)
    
    # Initialize models
    initialize_models()
    
    # Launch demo
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )
