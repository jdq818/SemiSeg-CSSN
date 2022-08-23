# SemiSeg-CSSN
Semi-supervised segmentation with cross supervision [flaree2022 challenge]
## Introduction
- SemiSeg-CSSN is an open source, PyTorch-based segmentation approach for 3D medical image. 
- SemiSeg-CSSN is adapted from EfficientSegmentation, please read the following paper:
[Efficient Context-Aware Network for Abdominal Multi-organ Segmentation](https://arxiv.org/abs/2109.10601).

## Features
- Cross supervision to train the SemiSeg-CSSN. Two networks with the same architecture were introduced, and they were initialized differently at the beginning of training. These two segmentation networks can generate pseudo label images,
and supervise each other’s training in the way of cross supervision. 
- We employed a filtering strategy (UIF) for unlabeled images. These selected unlabeled images have pseudo label images with low uncertainty, which can ensure the stability of training
- This method won the top 10 place on the [2022-MICCAI-FLARE](https://flare22.grand-challenge.org/) challenge. Where participants were required to effectively and efficiently segment multi-organ in abdominal CT with few labeled and large number of unlabeled images.
## Benchmark
|Task | L/U | DSC | NSC | Inference time(s) | GPU memory(MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
|[FLARE22](https://flare22.grand-challenge.org/)| 50/0 | 0.709 | 0.748 | - | - |
|[FLARE22](https://flare22.grand-challenge.org/)| 50/2000| 0.777 | 0.820 | 12.9 | 2052 |
|[FLARE22](https://flare22.grand-challenge.org/)| 50/200 (UIF)| 0.791 | 0.841 | - | - |

## Installation
#### Environment
- Ubuntu 20.04.4 LTS
- Python 3.6+
- Pytorch 1.5.0+
- CUDA 10.0+ 

1.Git clone
```
git clone 
```

2.Install Nvidia Apex
- Perform the following command:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

3.Install dependencies
```
pip install -r requirements.txt
```

## Get Started
### preprocessing
#### 
Reorientation, resampling and intensity normalization.
####
1. Download [FLARE22](https://flare22.grand-challenge.org/), resulting in 50 labeled images and masks, 2000 unlabeled images.
2. Copy image and mask to 'FlareSeg22/dataset/' folder.
3. Edit the 'FlareSeg22/data_prepare/config.yaml'. 
   'DATA_BASE_DIR'(Default: FlareSeg22/dataset/) is the base dir of databases.
   If set the 'IS_SPLIT_5FOLD'(Default: False) to true, 5-fold cross-validation datasets will be generated.
4. Run the data preprocess with the following command:
```bash
cd FlareSeg22/data_prepare
python run.py
```
The image data and lmdb file are stored in the following structure:
```wiki
DATA_BASE_DIR directory structure：
├── train_images
   ├── train_000_0000.nii.gz
   ├── train_001_0000.nii.gz
   ├── train_002_0000.nii.gz
   ├── ...
├── train_mask
   ├── train_000.nii.gz
   ├── train_001.nii.gz
   ├── train_002.nii.gz
   ├── ...
├── trainu_images
   ├── train_0000_0000.nii.gz
   ├── train_0001_0000.nii.gz
   ├── train_0002_0000.nii.gz
   ├── ...
└── val_images
    ├── validation_001_0000.nii.gz
    ├── validation_002_0000.nii.gz
    ├── validation_003_0000.nii.gz
    ├── ...
├── file_list
    ├──'train_series_uids.txt', 
    ├──'trainu_series_uids.txt', 
    ├──'val_series_uids.txt',
├── db
    ├──seg_raw_train         # The 50 labeled data information.
    ├──seg_raw_trainu         # The 2000 unlabeled data information.
    ├──seg_raw_test          # The validation images information.
    ├──seg_train_database    # The default training database.
    ├──seg_val_database      # The default validation database.
    ├──seg_pre-process_database # Temporary database.
    ├──segu_pre-process_database # Temporary database.
    ├──seg_train_fold_1
    ├──seg_val_fold_1
├── coarse_image
    ├──160_160_160
          ├── train_000.npy
          ├── train_001.npy
          ├── train_0000.npy
          ├── train_0001.npy
          ├── ...
├── coarse_mask
    ├──160_160_160
          ├── train_000.npy
          ├── train_001.npy
          ├── ...
├── fine_image
    ├──192_192_192
          ├── train_000.npy
          ├── train_001.npy
          ├── train_0000.npy
          ├── train_0001.npy
          ├──  ...
├── fine_mask
    ├──192_192_192
          ├── train_000.npy
          ├── train_001.npy
          ├── ...
```
The data information is stored in the lmdb file with the following format:
```wiki
{
    series_id = {
        'image_path': data.image_path,
        'mask_path': data.mask_path,
        'smooth_mask_path': data.smooth_mask_path,
        'coarse_image_path': data.coarse_image_path,
        'coarse_mask_path': data.coarse_mask_path,
        'fine_image_path': data.fine_image_path,
        'fine_mask_path': data.fine_mask_path
    }
}
```

### Models
- Models can be downloaded through [Baidu Netdisk](https://pan.baidu.com/s/1g-qyS2sXurqGcZ-2_ZF-fg), password:fl22
- Put the models in the "FlareSeg22/model_weights/" folder.


### Training
Remark: Coarse segmentation is trained on Nvidia V100(Number:16), while fine segmentation on Nvidia V100(Number:16). If you use different hardware, please set the "ENVIRONMENT.NUM_GPU", "DATA_LOADER.NUM_WORKER" and "DATA_LOADER.BATCH_SIZE" in 'FlareSeg22/coarse_base_seg/config.yaml' and 'FlareSeg22/fine_efficient_seg/config.yaml' files. You also need to set the 'nproc_per_node' in 'FlareSeg22/coarse_base_seg/run.sh' file.
#### Coarse segmentation:
- Edit the 'FlareSeg22/coarse_base_seg/config.yaml' and 'FlareSeg22/coarse_base_seg/run.sh'
- Train coarse segmentation with the following command:
```bash
cd FlareSeg22/coarse_base_seg
sh run.sh
```

#### Fine segmentation:
- Put the trained coarse model in the 'FlareSeg22/model_weights/base_coarse_model/' folder.
- Edit the 'FlareSeg22/fine_efficient_seg/config.yaml'.
- Edit the 'FlareSeg22/fine_efficient_seg/run.py', set the 'tune_params' for different experiments.
- Train fine segmentation with the following command:
```bash
cd  FlareSeg22/fine_efficient_seg
sh run.sh
```

### Inference:
- Put the trained models in the 'FlareSeg22/model_weights/' folder.
- Run the inference with the following command:
```bash
sh predict.sh
```

### Evaluation:
Refer to [FLARE2022 Evaluation](https://flare22.grand-challenge.org/Evaluation/).

## Contact
This repository is currently maintained by Jia Dengqiang (wangxifeng004@163.com).

## References
[1] Z. e. a. Zhu, “A 3d coarse-to-fine framework for volumetric medical image segmentation.” 2018 International Conference on 3D Vision (3DV), 2018.

[2] Q. e. a. Hou, “Strip pooling: Rethinking spatial pooling for scene parsing.” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[1] Zhang, F., Wang, Y., Yang, H.: Efficient context-aware network for abdominal
multi-organ segmentation. arXiv preprint arXiv:2109.10601 (2021)
[2] Chen, X., Yuan, Y., Zeng, G., Wang, J.: Semi-supervised semantic segmentation
with cross pseudo supervision. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. pp. 2613–2622 (2021)

[3] Yang, L., Zhuo, W., Qi, L., Shi, Y., Gao, Y.: St++: Make self-training work better for semi-supervised semantic segmentation. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. pp. 4268–4277 (2022)
## Acknowledgement
Thanks for FLARE organizers with the donation of the dataset.
