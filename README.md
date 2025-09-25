# SRP (Spectral Robustness Project) - User Guide

## Hardware Setup
NVIDIA GeForce RTX 3090 × 4

## Environment Setup

### Create Conda Environment
```
conda create --name SRP python=3.10
conda activate SRP
pip install -r requirements.txt
```

### Prepare Datasets
- **CIFAR-10**: Automatically downloaded to `SRP/data/`.
- **MNIST**: Automatically downloaded to `SRP/data/`.
- **Tiny-ImageNet**: Download from [tjmoon0104/pytorch-tiny-imagenet](https://github.com/tjmoon0104/pytorch-tiny-imagenet). Unzip to `SRP/data/tiny-imagenet/`.

## Training Models

### Command Format
```
python train.py --data {cifar10,mnist,tiny-imagenet} \
               --model {vit_b_16,resnet18,vgg16,densenet121} \
               --attack {cw,pgd,none}
```

### Example
```
python train.py --data cifar10 --model vit_b_16 --attack cw
```

### Output Files
Saved in `models/`:
- `[model]_[dataset]_[attack]_final.pth`
- `[model]_[dataset]_[attack]_best.pth`

**NOTE**: Training is required before evaluation!

## Model Evaluation

### Command Format
```
python main.py --eval {lips,clever,spec,cw,pgd} \
              --data {cifar10,mnist,tiny-imagenet} \
              --model {vit_b_16,resnet18,vgg16,densenet121} \
              --attack {cw,pgd,none} \
              --seed_n {0,1,2,3}
```

### Example
```
python main.py --eval lips --data cifar10 --model vit_b_16 --attack cw --seed_n 1
```

### Results
Saved in `cw-results-5/` (default folder; name may vary).

## Troubleshooting
- **Dataset download issues**: Check network or manually download Tiny-ImageNet.
- **GPU out of memory**: Reduce batch size in code.
- **Inconsistent results**: Use fixed random seed (`--seed_n` parameter).

## License
MIT License - see [LICENSE](LICENSE) file.
