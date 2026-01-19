from setuptools import setup, find_packages
import subprocess
import os


setup(
    name="m2m",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "accelerate>=1.9.0",
        "diffusers>=0.31.0",
        "opencv-python>=4.9.0.80",
        "tokenizers>=0.20.3",
        "transformers>=4.49.0",
        "numpy>=1.23.5,<2",
        # "xformers==0.0.24",
        "psutil",
        "flash_attn",
        "opencv-python>=4.9.0.80",
        "matplotlib",
        "scipy",
        "setuptools",
        "omegaconf",
        "tabulate",
        "pandas",
        "wandb",
        "datasets",
        "peft>0.15",
        "easydict",
        "boto3>=1.37.22",
        "ftfy",
        "scikit-image",
        "utils3d",
        "moviepy",
        "pyarrow",
        "imageio",
        "pycocotools",
        "deepspeed==0.17.1",
        # "f3fs"
    ],
    dependency_links=[
        "git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d"
    ],
    python_requires=">=3.9",
)