import sys
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_svd_mask import VideoInferencePipeline


@pytest.mark.parametrize("mask_cond_mode,in_channels", [("interpolate", 9), ("vae", 12)])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
def test_run_with_grayscale_mask(tmp_path, mask_cond_mode, in_channels, weight_dtype):
    """Run the public pipeline with small local models, without downloading weights."""
    torch.manual_seed(0)
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(
        base / "feature_extractor"
    )
    CLIPVisionModelWithProjection(CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, image_size=32, patch_size=16, projection_dim=32,
    )).save_pretrained(base / "image_encoder", variant="fp16")
    AutoencoderKLTemporalDecoder(
        block_out_channels=(32, 32), down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        layers_per_block=1,
    ).save_pretrained(base / "vae", variant="fp16")
    UNetSpatioTemporalConditionModel(
        in_channels=in_channels, out_channels=4, block_out_channels=(32,),
        down_block_types=("CrossAttnDownBlockSpatioTemporal",),
        up_block_types=("CrossAttnUpBlockSpatioTemporal",), layers_per_block=1,
        cross_attention_dim=32, num_attention_heads=(4,),
        addition_time_embed_dim=8, projection_class_embeddings_input_dim=24,
    ).save_pretrained(checkpoint / "unet")
    pipeline = VideoInferencePipeline(str(base), str(checkpoint), device="cpu", weight_dtype=weight_dtype)
    frames = [Image.new("RGB", (32, 32), "red") for _ in range(2)]
    masks = [Image.new("L", (32, 32), 255) for _ in range(2)]

    result = pipeline.run(frames, masks, mask_cond_mode=mask_cond_mode)

    assert len(result) == 2
    assert all(frame.size == (32, 32) and frame.mode == "RGB" for frame in result)
