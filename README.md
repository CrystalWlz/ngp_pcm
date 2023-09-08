
# OmniObject3D Challenge @ ICCV 2023
* Hosted by AI for 3D Content Creation Workshop
  
Track-2 | 3D Object Generation
This phase evaluates algorithms for realistic 3D object generation on the OmniObject3D dataset. Submit the post-processed results on the generated objects in a .zip files. Please refer to the tools and provided examples here and carefully check the format to ensure a successful submission.
# ngp_pcm
Instant-ngp (only NeRF) in pytorch+cuda trained with pytorch-lightning (**high quality with high speed**). This repo aims at providing a concise pytorch interface to generate a large number of object models and facilitate future research. 

*  [Official CUDA implementation](https://github.com/NVlabs/instant-ngp/tree/master)
*  [torch-ngp](https://github.com/ashawkey/torch-ngp) another pytorch implementation that I highly referenced.
*  [ngp_pl](https://github.com/kwea123/ngp_pl) a pytorch implementation that I highly referenced.


# Installation

This implementation has **strict** requirements due to dependencies on other libraries, if you encounter installation problem due to hardware/software mismatch, I'm afraid there is **no intention** to support different platforms (you are welcomed to contribute).

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB (Tested with RTX 2080 Ti), CUDA 11.3 (might work with older version)
* 32GB RAM (in order to load full size images)

## Software

* Clone this repo by `git clone https://github.com/kwea123/ngp_pl`
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ngp_pl python=3.8` to create a conda environment and activate it by `conda activate ngp_pl`)
* Python libraries
    * Install pytorch by `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation) 
    `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0%2Bcu113.html`
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension) `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)`cd apex`     
    *  `pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./`
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)
* Bugs:
    *  File "/data2_12t/wlz/anaconda3/envs/ngp_pl/lib/python3.8/site-packages/pytorch_lightning/callbacks/progress/rich_progress.py", line 20, in <module>from torchmetrics.utilities.imports import _compare_version,ImportError: cannot import name '_compare_version' from 'torchmetrics.utilities.imports' (/data2_12t/wlz/anaconda3/envs/ngp_pl/lib/python3.8/site-packages/torchmetrics/utilities/imports.py)只要把rich_progress.py里_compare_version改成compare_version
    * 
# Supported Datasets

1.  NSVF data

Download preprocessed datasets (`Synthetic_NeRF`, `Synthetic_NSVF`, `BlendedMVS`, `TanksAndTemples`) from [NSVF](https://github.com/facebookresearch/NSVF#dataset). **Do not change the folder names** since there is some hard-coded fix in my dataloader.

2.  NeRF++ data

Download data from [here](https://github.com/Kai-46/nerfplusplus#data).

3.  Colmap data

For custom data, run `colmap` and get a folder `sparse/0` under which there are `cameras.bin`, `images.bin` and `points3D.bin`. The following data with colmap format are also supported:

  *  [nerf_llff_data](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing) 
  *  [mipnerf360 data](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
  *  [HDR-NeRF data](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I). Additionally, download my colmap pose estimation from [here](https://drive.google.com/file/d/1TXxgf_ZxNB4o67FVD_r0aBUIZVRgZYMX/view?usp=sharing) and extract to the same location.

4. RTMV data

Download data from [here](http://www.cs.umd.edu/~mmeshry/projects/rtmv/). To convert the hdr images into ldr images for training, run `python misc/prepare_rtmv.py <path/to/RTMV>`, it will create `images/` folder under each scene folder, and will use these images to train (and test).

# Training

Quickstart: `python train.py --root_dir <path/to/lego> --exp_name Lego` 
//`python train.py --root_dir /data2_12t/dataset/Synthetic_NeRF/Lego/ --exp_name Lego`
//`python train.py --root_dir /data2_12t/dataset/3D_Project/ --exp_name test --dataset_name colmap --num_gpus 2`

It will train the Lego scene for 30k steps (each step with 8192 rays), and perform one testing at the end. The training process should finish within about 5 minutes (saving testing image is slow, add `--no_save_test` to disable). Testing PSNR will be shown at the end.

More options can be found in [opt.py](opt.py).

For other public dataset training, please refer to the scripts under `benchmarking`.

# Testing

Use `test.ipynb` to generate images. Lego pretrained model is available [here](https://github.com/kwea123/ngp_pl/releases/tag/v1.0)

GUI usage: run `python show_gui.py` followed by the **same** hyperparameters used in training (`dataset_name`, `root_dir`, etc) and **add the checkpoint path** with `--ckpt_path <path/to/.ckpt>`


# Benchmarks

To run benchmarks, use the scripts under `benchmarking`.


`python benchmarking/fid_score.py /data2_12t/dataset/OpenXD-OmniObject3D-New/results_img --reso 128 --save_path ./my_results`

