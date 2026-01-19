---
title: VideoMaMa - Video Matting with Mask Guidance
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎬 VideoMaMa: Video Matting with Mask Guidance

An interactive demo for high-quality video matting using sparse mask guidance. This demo combines SAM2 for automatic object tracking with our VideoMaMa model for generating alpha mattes.

## 🌟 Features

- **Single-Click Object Selection**: Simply click on the object you want to extract in the first frame
- **Automatic Tracking**: SAM2 automatically tracks your selected object through all frames
- **High-Quality Matting**: VideoMaMa generates smooth, temporally-consistent alpha mattes
- **Flexible Input**: Upload your own video or try our provided samples
- **Customizable**: Adjust augmentation settings for different scenarios

## 🚀 How to Use

1. **Upload a video** or **select from samples**
2. **Click on the object** you want to extract in the first frame (displayed in the interface)
3. Optionally adjust **augmentation settings** in the advanced options
4. Click **"Generate Matting"** and wait for processing
5. View your results: output video, comparison images, and mask track


## 🔧 Installation (Local Setup)

If you want to run this demo locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Add sample videos to samples/ directory (optional)

# Run the demo
python app.py
```

## 🎯 Tips for Best Results

- **Click Precisely**: Click on the center of the object you want to extract
- **Clear Objects**: Works best with distinct foreground objects
- **Video Length**: For faster processing, use shorter videos (< 5 seconds)
- **Augmentations**: 
  - Use "polygon" for cleaner geometric masks
  - Enable temporal augmentation for challenging videos
  - Try "bounding box" for very simple selections

## 📚 Technical Details

### Model Architecture
- **Base Model**: Stable Video Diffusion (SVD-XT)
- **Conditioning**: RGB frames + VAE-encoded masks
- **UNet**: Fine-tuned with additional mask conditioning channels
- **Processing**: Chunked inference (16 frames per chunk)

### SAM2 Integration
- Uses SAM2 video predictor for mask tracking
- Propagates mask from single click point through entire video
- Generates temporally consistent segmentation masks

## 🤝 Contributing

If you encounter issues or have suggestions:
1. Check that all model checkpoints are correctly placed
2. Ensure your GPU has sufficient VRAM
3. Try reducing video length or resolution for testing


## 🙏 Acknowledgments

- **SAM2**: Meta AI's Segment Anything 2
- **Stable Video Diffusion**: Stability AI's video generation model
- **Gradio**: For the amazing UI framework

## 📧 Contact

For questions or issues, please open an issue on our GitHub repository.

---

**Note**: This demo is for research purposes. Processing times may vary based on video length and available compute resources.
