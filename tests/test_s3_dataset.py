import io
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
from botocore.response import StreamingBody
from botocore.stub import Stubber
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import s3_utils
from dataloader.synthetic import AdobeVideoDataset
from dataloader.vos import VideoObjectSegmentationDataset


def test_dataset_reads_s3_metadata_and_frames(monkeypatch):
    client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
    monkeypatch.setattr(s3_utils, "get_s3_client", lambda: client)
    metadata = io.BytesIO()
    pd.DataFrame([{"composite_path": "images/000000", "alpha_path": "alphas/000000/0", "num_frames": 1}]).to_parquet(metadata, index=False)
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(image, format="JPEG")
    alpha = io.BytesIO()
    Image.new("L", (64, 64), 255).save(alpha, format="PNG")
    with Stubber(client) as stub:
        for key, data in [("metadata.parquet", metadata.getvalue()), (os.path.join("images/000000", "0000.jpg"), image.getvalue()), (os.path.join("alphas/000000/0", "0000.png"), alpha.getvalue())]:
            stub.add_response("get_object", {"Body": StreamingBody(io.BytesIO(data), len(data))}, {"Bucket": "fixture-bucket", "Key": key})
        dataset = AdobeVideoDataset("fixture-bucket", "", "metadata.parquet", num_frames=1, height=64, width=64, temporal_augmentation_rate=0)
        sample = dataset[0]
        assert sample["video"].shape == sample["alpha"].shape == sample["binary_mask"].shape == (1, 3, 64, 64)
        assert sample["binary_mask"].min() == 1
        stub.assert_no_pending_responses()


def test_local_vos_dataset(tmp_path):
    images = tmp_path / "JPEGImages" / "clip"
    masks = tmp_path / "Annotations" / "clip"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.new("RGB", (64, 64), "red").save(images / "0000.jpg")
    Image.new("P", (64, 64), 1).save(masks / "0000.png")
    sample = VideoObjectSegmentationDataset(str(tmp_path), num_frames=2, height=64, width=64, temporal_augmentation_rate=0)[0]
    assert sample["video"].shape == sample["alpha"].shape == (2, 3, 64, 64)
    assert sample["binary_mask"].min() == 1
