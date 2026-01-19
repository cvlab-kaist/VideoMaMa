#!/bin/bash
# Download model checkpoints for VideoMaMa demo

set -e

echo "🔽 Downloading model checkpoints for VideoMaMa demo..."
echo ""

# Create checkpoints directory
echo "Creating checkpoints directory..."
mkdir -p checkpoints
echo "✓ Directory created"
echo ""

# Download SAM2 checkpoint
echo "Downloading SAM2 checkpoint..."
echo "URL: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
echo "This may take a few minutes (file size: ~900MB)..."

if command -v wget &> /dev/null; then
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
         -O checkpoints/sam2_hiera_large.pt
elif command -v curl &> /dev/null; then
    curl -L https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
         -o checkpoints/sam2_hiera_large.pt
else
    echo "❌ Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

echo "✓ SAM2 checkpoint downloaded successfully"
echo ""

# Check if VideoMaMa checkpoint exists
echo "Checking VideoMaMa checkpoint..."
if [ -d "checkpoints/videomama_unet" ]; then
    if [ -f "checkpoints/videomama_unet/config.json" ] && \
       { [ -f "checkpoints/videomama_unet/diffusion_pytorch_model.safetensors" ] || \
         [ -f "checkpoints/videomama_unet/diffusion_pytorch_model.bin" ]; }; then
        echo "✓ VideoMaMa checkpoint already exists"
    else
        echo "⚠️  VideoMaMa checkpoint directory exists but is incomplete"
        echo "   Please add the following files to checkpoints/videomama_unet/:"
        echo "   - config.json"
        echo "   - diffusion_pytorch_model.safetensors (or .bin)"
    fi
else
    echo "⚠️  VideoMaMa checkpoint not found"
    echo ""
    echo "📝 Manual step required:"
    echo "   1. Create directory: checkpoints/videomama_unet/"
    echo "   2. Copy your trained VideoMaMa checkpoint files:"
    echo "      - config.json"
    echo "      - diffusion_pytorch_model.safetensors (or .bin)"
    echo ""
    echo "   Example:"
    echo "   mkdir -p checkpoints/videomama_unet"
    echo "   cp /path/to/your/checkpoint/* checkpoints/videomama_unet/"
fi

echo ""
echo "="*70
echo "✨ Checkpoint download complete!"
echo "="*70
echo ""
echo "Next steps:"
echo "1. Verify checkpoints are in place:"
echo "   python test_setup.py"
echo ""
echo "2. (Optional) Add sample videos:"
echo "   mkdir -p samples"
echo "   cp your_sample.mp4 samples/"
echo ""
echo "3. Test locally:"
echo "   python app.py"
echo ""
echo "4. Deploy to Hugging Face Space"
echo ""
