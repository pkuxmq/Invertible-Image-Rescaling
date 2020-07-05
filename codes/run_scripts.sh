# single GPU training
python train.py -opt options/train/train_IRN_x4.yml

# distributed training
# 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/train_IRN_x4.yml --launcher pytorch

