# Mixup-Inference-in-Training

This is the implementation of our ICML'24 paper [Calibration Bottleneck: Over-compressed Representations are Less Calibratable](https://dengbaowang.github.io/). In the paper, we observe a U-shaped pattern in the calibratability of intermediate features, spanning from the lower to the upper layers.

## Dependencies
This code requires the following:

* Python 3.8, 
* numpy 1.24.3, 
* Pytorch 1.13.1+cu116, 
* torchvision 0.14.1+cu116.

## Training
For example, you can:

1. Download SVHN/CIFAR-10/CIFAR-100/Tiny-ImageNet dataset into `../data/`.

2. Run the following demos:
```
python train.py  --dataset cifar100 --arch resnet18 --weight-decay 5e-4 --seed 101 --gamma 1.0 --PLP # Using PLP

python train.py  --dataset cifar100 --arch resnet18 --weight-decay 5e-4 --seed 101 --gamma 1.0 # No PLP

python train_tiny.py --dataset tinyimagenet --model resnet18  --seed_val 101 --epochs 200 --lr 0.01 --bs 64 --wd 1e-4 --seed 123 --nw 16 --pretrained --gamma 1.0 --PLP # Using PLP

python train_tiny.py --dataset tinyimagenet --model resnet18  --seed_val 101 --epochs 200 --lr 0.01 --bs 64 --wd 1e-4 --seed 123 --nw 16 --pretrained --gamma 1.0 # No PLP
```

## Citation
```
@inproceedings{ICML2024wang,
author = {Deng-Bao Wang, Min-Ling Zhang},
title = {Calibration Bottleneck: Over-compressed Representations are Less Calibratable},
booktitle = {Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria.},
year = {2024}
}
```

## Contact
If you have any further questions, please feel free to send an e-mail to: wangdb@seu.edu.cn.
