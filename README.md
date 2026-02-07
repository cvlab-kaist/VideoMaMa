<div align="center">
<h1>VideoMaMa: Mask-Guided Video Matting via Generative Prior</h1>

[**Sangbeom Lim**](https://sites.google.com/view/sangbeomlim/home)<sup>1</sup> · [**Seoung Wug Oh**](https://sites.google.com/view/seoungwugoh)<sup>2</sup> · [**Jiahui Huang**](https://gabriel-huang.github.io/)<sup>2</sup> · [**Heeji Yoon**](https://yoon-heez.github.io/
)<sup>3</sup>  
[**Seungryong Kim**](https://cvlab.kaist.ac.kr/members/faculty)<sup>3</sup> · [**Joon-Young Lee**](https://joonyoung-cv.github.io)<sup>2</sup>

<sup>1</sup>Korea University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Adobe Research&emsp;&emsp;&emsp;&emsp;<sup>3</sup>KAIST AI

**ArXiv 2026**

<a href="https://arxiv.org/abs/2601.14255"><img src='https://img.shields.io/badge/arXiv-VideoMaMa-red' alt='Paper PDF'></a>
<a href='https://cvlab-kaist.github.io/VideoMaMa/'><img src='https://img.shields.io/badge/Project_Page-VideoMaMa-green' alt='Project Page'></a>
<a href="https://huggingface.co/spaces/SammyLim/VideoMaMa" target='_blank'>
<img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue">
</a>

<strong>VideoMaMa is a mask-guided video matting framework that leverages a video generative prior. By utilizing this prior, it supports stable performance across diverse video domains with fine-grained matting quality.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/teaser.jpg">
</div>

For more visual results, go checkout our <a href="https://cvlab-kaist.github.io/VideoMaMa/" target="_blank">project page</a>
</div>

## 📰 News
VideoMaMa is an open-source project. If you find our work helpful, please consider giving this repository a ⭐.  

* **2026-01-19:** Our [Github Repo](https://github.com/cvlab-kaist/VideoMaMa) is opened!
* **2026-02-07** [ComfyUI-VideoMaMa](https://github.com/okdalto/ComfyUI-VideoMaMa) is now available! (Thanks to [@okdalto](https://github.com/okdalto))

**Note: Training code is currently under internal review. Release coming soon.**


## 🔥 TODO
- [x] Release Demo & Model checkpoint. (Jan 19, 2025)
- [x] Release ArXiv paper. (Jan 19, 2025)
- [ ] Release Training Code.
- [ ] Evaluation Code.
- [ ] Release MA-V dataset.

# ⚙️ Setup
Please run
```bash
bash scripts/setup.sh
```
it will down load stable video diffusion weight, and setup virtual enviroment needed to run whole codes.  
We use `conda activate videomama`.

This will download sam2 which is needed for training sam2-matte.


# 🎮 Demo
Please check [demo readme](demo/README.md).

# 🎯 Inference
[![Hugging Face Model Card](https://img.shields.io/badge/Hugging%20Face-VideoMaMa-orange?logo=huggingface&style=flat-square)](https://huggingface.co/SammyLim/VideoMaMa)

VideoMaMa model checkpoint — available on the Hugging Face Hub: [SammyLim/VideoMaMa](https://huggingface.co/SammyLim/VideoMaMa).

For inferencing video use this command.
```bash
python inference_onestep_folder.py \
--base_model_path "<stabilityai/stable-video-diffusion-img2vid-xt_path>" \
--unet_checkpoint_path "<videomama_checkpoint_path>" \
--image_root_path "/assets/example/image" \
--mask_root_path "assets/example/mask" \
--output_dir "assets/example" \
[--optional_arguments]
```

For example, If you have setup using above command, this example bash will work.
```bash
python inference_onestep_folder.py \
    --base_model_path "checkpoints/stable-video-diffusion-img2vid-xt" \
    --unet_checkpoint_path "checkpoints/VideoMaMa" \
    --image_root_path "/assets/example/image" \
    --mask_root_path "assets/example/mask" \
    --output_dir "assets/example" \
    --keep_aspect_ratio 
```

For more information about inference setting, please check [inference readme](inference.md).

# 🚂🚃🚃🚃🚃 Training

### Generating training dataset
Please check [Data pipeline README](data_pipeline/data_pipeline.md).

### Model Training
Please check [training README](training.md).

<!-- # Evaluation
Please check [evaluation readme](evaluation/README.md). -->

# 🎓 Citation
```
@article{lim2026videomama,
  title={VideoMaMa: Mask-Guided Video Matting via Generative Prior},
  author={Lim, Sangbeom and Oh, Seoung Wug and Huang, Jiahui and Yoon, Heeji and Kim, Seungryong and Lee, Joon-Young},
  journal={arXiv preprint arXiv:2601.14255},
  year={2026}
}
```


## 🙏 Acknowledgments
- **SAM2**: Meta AI's Segment Anything 2
- **Stable Video Diffusion**: Stability AI's video generation model
- **Gradio**: For the amazing UI framework

## 📧 Contact
For questions or issues, please open an issue on our GitHub repository.

We welcome any feedback, questions, or opportunities for collaboration. If you are interested in using this model for industrial applications, or have specific questions about the architecture and training, please feel free to reach out.

## 📄 License
The code in this repository is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license, unless otherwise specified.

This repository builds on implementations and ideas from the [Hugging Face](https://huggingface.co) ecosystem and the [diffusion-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft) project. Many thanks to the original authors and contributors for their open-source work.

The VideoMaMa model checkpoints (specifically `VideoMama/unet/*` and `dino_projection_mlp.pth`) are subject to the [Stability AI Community License](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/LICENSE.md#:~:text=%22Agreement%22%20means%20this%20Stability%20AI,updated%20from%20time%20to%20time).