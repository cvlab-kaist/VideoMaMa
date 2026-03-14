# Version 2

To generate synthetic dataset, we follow two step.
1. Generate synthetic dataset by composing foreground and background.
2. Upload folder to s3.

### Data generation
We should set the paths of our foregrounds and background that to be used as a dataset.  
On `config/fg_alpha_bg_paths.yaml`, you can set data source paths.
```

Below is the example of making synthetic dataset.
```bash
python data_pipeline/generate_synthetic_v2.py --config-file data_pipeline/config/fg_alpha_bg_paths.yaml \
--output-dir <target_path>/video_synthetic \
--max-num-videos 30000 --max-instances 2 --min-instances 1 \
--num-frames 10 --n-workers 64 --instance-ratio 0:3 --save-mode "files" --fg-scale-range 1.2,1.5
```