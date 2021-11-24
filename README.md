# This repository contains supplementary code for anonymous submission ``_Mitigating the Data Bottleneck in Unpaired Image-to-Image Translation with Synthetic Imagery_''

## Inference

Download a dataset

```bash ./datasets/download_cut_dataset.sh horse2zebra```

Download and unpack pretrained models

```wget https://www.dropbox.com/sh/ -O checkpoints.tar && tar -xvf checkpoints.tar && rm checkpoints.tar```

Run ```sampling.ipynb``` for translation examples and synthetic-init samples.

## Training

This code will create a local directory ```./runs/horse2zebra_cut_with_aug``` and start CUT model training with augmentations and initialization from ```runs/horse2zebra_cut_with_aug/checkpoint.pt``` (if existed).

```python train.py --dataroot datasets/horse2zebra --name horse2zebra --CUT_mode CUT --save_epoch_freq 5 --out_dir runs/horse2zebra_cut_with_aug --checkpoint_path runs/horse2zebra_cut_with_aug/checkpoint.pt --drop_checkpoint_epochs True --load_state_from_checkpoint True --n_epochs 400 --n_epochs_decay 0 --ada_stop_epoch 400 --ada_p 0.2 --ada_fixed True```

## Credits
* This code is based on the original CUT and CycleGAN realization:
https://github.com/taesungp/contrastive-unpaired-translation

* We actively utilize the original StyleGAN-ADA:
https://github.com/NVlabs/stylegan2-ada

* FID calculation is based on:
https://github.com/mseitzer/pytorch-fid

to make the code self-sufficient, we add these projects as subdirectories.