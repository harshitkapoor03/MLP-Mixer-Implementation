
---

# 🧩 MLP-Mixer for CIFAR-10 (PyTorch Implementation)

A PyTorch implementation of the **MLP-Mixer** architecture for image classification on the CIFAR-10 dataset.  
This project re-creates the ideas presented in [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) using modern training practices.

---
Note:
This code is implemented for the CIFAR-10 dataset but can easily be adapted for any image classification dataset by adjusting the training hyperparameters.
It's important to note that, conventionally, MLP-Mixer models perform poorly on small datasets like CIFAR-10 when trained from scratch, and their strengths become more apparent at larger scales (e.g., ImageNet). Nevertheless, due to computational constraints, this project focuses on CIFAR-10. Despite the limitations, the model achieves respectable accuracy and loss values compared to convolutional and transformer-based models when trained properly.


## 📑 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How It Works (Relation to Paper)](#how-it-works-relation-to-paper)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Applications](#applications)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

The project implements an MLP-only image classification model that:
- **Replaces convolution** and **attention** layers with pure MLPs.
- **Separately mixes** spatial (token) and feature (channel) information.
- **Uses modern training tricks** like RandAugment, Mixup, label smoothing, and a cosine learning rate scheduler.
  
Our target dataset is **CIFAR-10**, and we train the model fully from scratch, following the philosophy that convolutions and self-attention are *not necessary* for achieving good vision model performance.

---

## 🚀 Features

- ✅ **Pure MLP architecture** with token-mixing and channel-mixing MLPs  
- 📦 **RandAugment** and **Mixup** data augmentation  
- 🧹 **Label smoothing** in loss function  
- 📉 **Warmup + CosineAnnealing** learning rate scheduling  
- 🧪 **Early stopping** based on validation accuracy (manual save)  
- 📈 Training visualization for accuracy and loss curves  

---

## 📚 How It Works (Relation to Paper)

This codebase is a faithful implementation of key ideas from the MLP-Mixer paper :

| Paper Concept                         | Our Implementation                                  |
|:---------------------------------------|:----------------------------------------------------|
| **Patch Embedding**                    | Conv2D layer with stride = patch size               |
| **Flatten patches into tokens**        | Flatten + rearrange (`B, C, H, W -> B, N_tokens, C`) |
| **Token-Mixing MLP**                   | `MlpBlock` after LayerNorm and transposing tokens   |
| **Channel-Mixing MLP**                 | `MlpBlock` after LayerNorm (row-wise)               |
| **GELU activation**                    | `nn.GELU` in MLP blocks                             |
| **Residual (Skip) connections**        | Add original input after each MLP block             |
| **Layer Normalization**                | Before each MLP block                              |
| **Global Average Pooling**             | Mean over tokens before final head                  |
| **Final Head**                         | Linear layer to project to class logits             |
| **Stochastic Depth (DropPath)**         | Implemented manually inside `MixerBlock`            |

**Training Enhancements (Beyond the Paper):**
- **RandAugment** for stronger data augmentation.
- **Mixup** to prevent overfitting.
- **Label smoothing** to stabilize cross-entropy loss.
- **Linear Warmup + Cosine Annealing** for smoother learning rate schedules.
- **Random Erasing** for better regularization.

Thus, we stay true to the spirit of the MLP-Mixer:  
> *"Convolutions and self-attention are sufficient but not necessary for high performance in vision."* 

---

## 📁 Repository Structure

| File/Folder                         | Description                                                             |
|-------------------------------------|-------------------------------------------------------------------------|
| `mlp_mixer_cifar10.py`              | Full training script, including model, dataloaders, train/val loops     |
| `README.md`                         | Documentation file                                                      |


---

## ⚙️ Installation

### ✅ Prerequisites
- Python 3.7+
- PyTorch >= 1.10
- GPU recommended (for reasonable training time)

### 📦 Install required libraries:

```bash
pip install torch torchvision matplotlib numpy
```

---

## 🏁 Usage Guide

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/mlp-mixer-cifar10.git
cd mlp-mixer-cifar10
```

2. **Run the training script**:
```bash
python mlp_mixer_cifar10.py
```

3. **Outputs**:
   - Training and validation metrics will be printed every epoch.
   - Final test accuracy will be evaluated.
   - Accuracy and Loss curves will be plotted automatically.

---

## 🌍 Applications

- Research on **pure MLP** vision architectures.
- Educational demo for replacing **CNNs/Transformers**.
- Lightweight models for **embedded** or **low-power** vision applications.

---

## 🤝 Contributing

Pull requests are welcome!  
Feel free to open an issue for feature requests, improvements, or bug reports.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

