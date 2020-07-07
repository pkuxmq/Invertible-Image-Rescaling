# Invertible Image Rescaling
This is the PyTorch implementation of paper: [Invertible Image Rescaling](https://arxiv.org/abs/2005.05650).

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`
  
## Dataset Preparation
Commonly used training and testing datasets can be downloaded [here](https://github.com/xinntao/BasicSR/wiki/Prepare-datasets-in-LMDB-format).

## Get Started
Execution codes are in ['codes/'](https://github.com/mingqing/Invertible-Image-Rescaling/tree/master/codes/options).

### Training
First set a config file in options/train/, then execute as following:

	python train.py -opt options/train/train_IRN_x4.yml

### Test
First set a config file in options/test/, then execute as following:

	python test.py -opt options/test/test_IRN_x4.yml

Pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1K-DmWU7fO5Rf6EOmeW-8WEmhQQmX1pn7?usp=sharing).

## Acknowledgement
The code is based on [BasicSR](https://github.com/xinntao/BasicSR), with reference of [FrEIA](https://github.com/VLL-HD/FrEIA).

## Contact
If you have any questions, please contact <mingqing_xiao@pku.edu.cn>.
