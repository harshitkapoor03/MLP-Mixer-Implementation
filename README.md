# MLP-Mixer for CIFAR-10 (PyTorch Implementation)

This is a PyTorch implementation of the MLP-Mixer architecture from the paper ["MLP-Mixer: An all-MLP Architecture for Vision"](https://arxiv.org/abs/2105.01601), trained on CIFAR-10. I built this to better understand how pure MLP architectures perform compared to CNNs and Transformers for vision tasks.

## Key Features

- Pure MLP architecture (no convolutions or attention)
- Implements both token-mixing and channel-mixing MLPs
- Includes modern training enhancements:
  - RandAugment and Mixup data augmentation
  - Label smoothing
  - Cosine learning rate schedule with warmup
  - Early stopping based on validation accuracy
- Training visualization (accuracy/loss curves)

## Implementation Notes

The model follows the paper's architecture closely:
- Patch embedding via strided convolution
- Alternating token-mixing and channel-mixing MLP blocks
- Layer normalization before each MLP
- Skip connections around each MLP block
- Global average pooling before final classification head

I added several training improvements beyond the paper:
- RandAugment and Mixup for better regularization
- Label smoothing (0.1) to prevent overconfidence
- Cosine LR schedule with linear warmup
- Random erasing augmentation

## Performance

On CIFAR-10, the model achieves ~85% accuracy with default settings. This isn't state-of-the-art (CNNs/Transformers do better), but shows MLPs can work reasonably well for vision tasks.

The paper notes MLP-Mixers work better at larger scales (ImageNet), but I implemented this on CIFAR-10 due to compute constraints.

## Requirements

- Python 3.7+
- PyTorch 1.10+
- torchvision
- matplotlib, numpy

## Usage

1. Clone the repo:
```bash
git clone https://github.com/your-username/mlp-mixer-cifar10.git
cd mlp-mixer-cifar10
```

2. Run training:
```bash
python mlp_mixer_cifar10.py
```

The script will:
- Download CIFAR-10 automatically
- Train the model with progress logging
- Plot training/validation curves
- Report final test accuracy

## Customization

You can tweak:
- Model dimensions (embed dim, MLP sizes)
- Training hyperparameters (LR, batch size)
- Augmentation strength
- Number of epochs

## Why This Project?

I was curious if MLPs could really compete with CNNs for vision tasks. This implementation helped me understand:
- How token mixing works compared to convolutions
- The importance of good normalization and residual connections
- How much data augmentation matters for MLP architectures

## Limitations

- Performance is modest on small datasets
- Training is slower than CNNs (no optimized conv2d)
- Not as parameter-efficient as modern architectures

## Contributing

Feel free to open issues or PRs if you have improvements! Some ideas:
- Add distributed training support
- Implement more efficient MLP variants
- Port to other datasets


