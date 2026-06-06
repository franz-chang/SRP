# SRP (Spectral Robustness Project) - User Guide

## Abstract

The robustness of deep neural networks is crucial for safety-critical deployments, yet existing evaluation methods are often attack-dependent and lack interpretability. We propose a principled, attack-agnostic robustness metric based on the spectral norm of the Fisher Information Matrix (FIM), which quantifies the worst-case sensitivity of the model’s output distribution to input perturbations. Theoretically, we establish that the FIM equals the variance of the input Jacobian and derive closed-form spectral bounds for common architectures (VGG, ResNet, DenseNet, Transformer), providing the first theoretical robustness ranking. To enable scalable evaluation, we develop efficient algorithms—including power iteration and Hutchinson-based estimation—that support both white-box and black-box settings. Extensive experiments across multiple datasets (CIFAR, ImageNet, and medical images) and architectures show a strong correlation between our metric and adversarial vulnerability. Our framework serves as an interpretable diagnostic tool that complements attack-based evaluations, offering insights into architectural sensitivity and guiding the design of more robust models. 
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
