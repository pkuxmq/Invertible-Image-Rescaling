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
We use datasets in LDMB format for faster IO speed. Please refer to [wiki](https://github.com/xinntao/BasicSR/wiki/Prepare-datasets-in-LMDB-format) for more details.

## Get Started
Please see [wiki](https://github.com/xinntao/BasicSR/wiki/Training-and-Testing) for the basic usage, *i.e.,* training and testing.

## Model Zoo and Baselines
Results and pre-trained models are available in the [wiki-Model zoo](https://github.com/xinntao/BasicSR/wiki/Model-Zoo).
