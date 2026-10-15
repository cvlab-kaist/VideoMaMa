#!/bin/bash
# Download model checkpoints for VideoMaMa demo

set -e

echo "🔽 Downloading model checkpoints for VideoMaMa demo..."
echo ""

# Create checkpoints directory
echo "Creating checkpoints directory..."
mkdir -p checkpoints/sam2
echo "✓ Directory created"
echo ""

# Download SAM2 checkpoint
echo "Downloading SAM2 checkpoint..."
echo "URL: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
echo "This may take a few minutes (file size: ~900MB)..."

if command -v wget &> /dev/null; then
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
         -O checkpoints/sam2/sam2.1_hiera_large.pt
elif command -v curl &> /dev/null; then
    curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
         -o checkpoints/sam2/sam2.1_hiera_large.pt
else
    echo "❌ Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

echo "✓ SAM2 checkpoint downloaded successfully"
echo ""

# Check if VideoMaMa checkpoint exists
echo "Checking VideoMaMa checkpoint..."
if [ -d "checkpoints/VideoMaMa" ]; then
    if [ -f "checkpoints/VideoMaMa/unet/config.json" ] && \
       { [ -f "checkpoints/VideoMaMa/unet/diffusion_pytorch_model.safetensors" ] || \
         [ -f "checkpoints/VideoMaMa/unet/diffusion_pytorch_model.bin" ]; }; then
        echo "✓ VideoMaMa checkpoint already exists"
    else
        echo "⚠️  VideoMaMa checkpoint directory exists but is incomplete"
        echo "   Please add the following files to checkpoints/VideoMaMa/unet/:"
        echo "   - config.json"
        echo "   - diffusion_pytorch_model.safetensors (or .bin)"
    fi
else
    echo "⚠️  VideoMaMa checkpoint not found"
    echo ""
    echo "📝 Manual step required:"
    echo "   1. Create directory: checkpoints/VideoMaMa/unet/"
    echo "   2. Copy your trained VideoMaMa checkpoint files:"
    echo "      - config.json"
    echo "      - diffusion_pytorch_model.safetensors (or .bin)"
    echo ""
    echo "   Example:"
    echo "   mkdir -p checkpoints/VideoMaMa/unet"
    echo "   cp /path/to/your/checkpoint/unet/* checkpoints/VideoMaMa/unet/"
fi

echo ""
echo "="*70
echo "✨ Checkpoint download complete!"
echo "="*70
echo ""
echo "Next steps:"
echo "1. Verify checkpoints are in place under checkpoints/sam2/ and checkpoints/VideoMaMa/unet/."
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
