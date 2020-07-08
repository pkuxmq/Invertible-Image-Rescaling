# Training
First set a config file in options/train/, then run as following:

	python train.py -opt options/train/train_IRN_x4.yml

# Test
First set a config file in options/test/, then run as following:

	python test.py -opt options/test/test_IRN_x4.yml

Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1K-DmWU7fO5Rf6EOmeW-8WEmhQQmX1pn7?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1U38SjqVlqY5YVMsSFrkTsw) (extraction code: lukj).

# Code Framework
The code framework follows [BasicSR](https://github.com/xinntao/BasicSR/tree/master/codes). It mainly consists of four parts - `Config`, `Data`, `Model` and `Network`.

Let us take the train command `python train.py -opt options/train/train_IRN_x4.yml` for example. A sequence of actions will be done after this command. 

- [`train.py`](./train.py) is called. 
- Reads the configuration in [`options/train/train_IRN_x4.yml`](./options/train/train_IRN_x4.yml), including the configurations for data loader, network, loss, training strategies and etc. The config file is processed by [`options/options.py`](./options/options.py).
- Creates the train and validation data loader. The data loader is constructed in [`data/__init__.py`](./data/__init__.py) according to different data modes.
- Creates the model (is constructed in [`models/__init__.py`](./models/__init__.py) according to different model types). 
- Start to train the model. Other actions like logging, saving intermediate models, validation, updating learning rate and etc are also done during the training.  

## Contents

### Config
#### [`options/`](./options) Configure the options for data loader, network structure, model, training strategies and etc.

### Data
#### [`data/`](./data) A data loader to provide data for training, validation and testing.

### Model
#### [`models/`](./models) Construct models for training and testing.

### Network
#### [`models/modules/`](./models/modules) Construct different network architectures.

