# Invertible Image Rescaling
This is the PyTorch implementation of paper: Invertible Image Rescaling (ECCV 2020 Oral). [arxiv](https://arxiv.org/abs/2005.05650).

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
Training and testing codes are in ['codes/'](./codes/). Please see ['codes/README.md'](./codes/README.md) for basic usages.

## Invertible Architecture
![Invertible Architecture](./figures/architecture.jpg)

## Quantitative Results
Quantitative evaluation results (PSNR / SSIM) of different downscaling and upscaling methods for image reconstruction on benchmark datasets: Set5, Set14, BSD100, Urban100, and DIV2K validation set. For our method, differences on average PSNR / SSIM from different z samples are less than 0.02. We report the mean result over 5 draws.

| Downscaling & Upscaling | Scale | Param | Set5 | Set14 | BSD100 | Urban100 | DIV2K |
| :-----------------------: | :-----: | :-----: | :----: | :-----: | :------: | :--------: | :-----: |
| Bicubic & Bicubic       | 2x    | / | 33.66 / 0.9299 | 30.24 / 0.8688 | 29.56 / 0.8431 | 26.88 / 0.8403 | 31.01 / 0.9393 |
| Bicubic & SRCNN | 2x | 57.3K | 36.66 / 0.9542 | 32.45 / 0.9067 | 31.36 / 0.8879 | 29.50 / 0.8946 | – |
| Bicubic & EDSR | 2x | 40.7M | 38.20 / 0.9606 | 34.02 / 0.9204 | 32.37 / 0.9018 | 33.10 / 0.9363 | 35.12 / 0.9699 |
| Bicubic & RDN | 2x | 22.1M | 38.24 / 0.9614 | 34.01 / 0.9212 | 32.34 / 0.9017 | 32.89 / 0.9353 | – |
| Bicubic & RCAN | 2x | 15.4M |   38.27 / 0.9614   | 34.12 / 0.9216 | 32.41 / 0.9027 | 33.34 / 0.9384 | – |
| Bicubic & SAN | 2x | 15.7M | 38.31 / 0.9620 |   34.07 / 0.9213   |   32.42 / 0.9028   |   33.10 / 0.9370   |         –          |
|        TAD & TAU        |  2x   |   –   |     38.46 / –      |     35.52 / –      | 36.68 / – | 35.03 / – | 39.01 / – |
| CNN-CR & CNN-SR | 2x | – |     38.88 / –      | 35.40 / – | 33.92 / – | 33.68 / – | – |
| CAR & EDSR | 2x | 51.1M |   38.94 / 0.9658   | 35.61 / 0.9404 | 33.83 / 0.9262 | 35.24 / 0.9572 | 38.26 / 0.9599 |
| IRN (ours) | 2x | 1.66M | **43.99 / 0.9871** | **40.79 / 0.9778** | **41.32 / 0.9876** | **39.92 / 0.9865** | **44.32 / 0.9908** |

| Downscaling & Upscaling | Scale | Param |        Set5        |       Set14        |       BSD100       |      Urban100      |       DIV2K        |
| :---------------------: | :---: | :---: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|    Bicubic & Bicubic    |  4x   |   /   |   28.42 / 0.8104   |   26.00 / 0.7027   |   25.96 / 0.6675   |   23.14 / 0.6577   |   26.66 / 0.8521   |
|     Bicubic & SRCNN     |  4x   | 57.3K |   30.48 / 0.8628   |   27.50 / 0.7513   |   26.90 / 0.7101   |   24.52 / 0.7221   |         –          |
|     Bicubic & EDSR      |  4x   | 43.1M |   32.62 / 0.8984   |   28.94 / 0.7901   |   27.79 / 0.7437   |   26.86 / 0.8080   |   29.38 / 0.9032   |
|      Bicubic & RDN      |  4x   | 22.3M |   32.47 / 0.8990   |   28.81 / 0.7871   |   27.72 / 0.7419   |   26.61 / 0.8028   |         –          |
|     Bicubic & RCAN      |  4x   | 15.6M |   32.63 / 0.9002   |   28.87 / 0.7889   |   27.77 / 0.7436   |   26.82 / 0.8087   |   30.77 / 0.8460   |
|    Bicubic & ESRGAN     |  4x   | 16.3M |   32.74 / 0.9012   |   29.00 / 0.7915   |   27.84 / 0.7455   |   27.03 / 0.8152   |   30.92 / 0.8486   |
|      Bicubic & SAN      |  4x   | 15.7M |   32.64 / 0.9003   |   28.92 / 0.7888   |   27.78 / 0.7436   |   26.79 / 0.8068   |         –          |
|        TAD & TAU        |  4x   |   –   |     31.81 / –      |     28.63 / –      |     28.51 / –      |     26.63 / –      |     31.16 / –      |
|       CAR & EDSR        |  4x   | 52.8M |   33.88 / 0.9174   |   30.31 / 0.8382   |   29.15 / 0.8001   |   29.28 / 0.8711   |   32.82 / 0.8837   |
|       IRN (ours)        |  4x   | 4.35M | **36.19 / 0.9451** | **32.67 / 0.9015** | **31.64 / 0.8826** | **31.41 / 0.9157** | **35.07 / 0.9318** |



## Qualitative Results
![Qualitative results of upscaling the 4x downscaled images](./figures/qualitative_results.jpg)

## Acknowledgement
The code is based on [BasicSR](https://github.com/xinntao/BasicSR), with reference of [FrEIA](https://github.com/VLL-HD/FrEIA).

## Contact
If you have any questions, please contact <mingqing_xiao@pku.edu.cn>.
