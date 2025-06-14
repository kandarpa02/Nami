# Nami: A State-of-the-Art Activation Function for Deep Convolutional Networks

## Overview

**Nami** is a novel activation function designed to outperform widely used non-linearities such as **Mish** and **Swish** in training deep convolutional networks. This study demonstrates Nami's superior performance on the CIFAR-100 classification task using a ResNet-50 backbone, mixed-precision training, and an SGD optimizer with learning rate scheduling.

- **Architecture**: ResNet-50  
- **Dataset**: CIFAR-100  
- **Training Epochs**: 100  
- **Optimizer**: SGD with learning rate 0.1  
- **Precision**: Mixed Precision Training (AMP)  
- **Scheduler**: CosineAnnealingLR  
- **Batch Size**: 128  

## Activation Function Derivative

<div align="center">
  <img src="benchmarks/results1/Nami_derivative.png" style="width:80%;" />
</div>

Nami's formulation allows for a smooth and bounded derivative, enabling more stable backpropagation and faster convergence compared to Mish and Swish.

---

## Performance Comparison

### Top-1 Validation Accuracy

- **Nami**: **67.28%**
- **Mish**: 66.61%
- **Swish**: 65.10% (peak around epoch 93)

Nami consistently reaches higher top-1 accuracy earlier and maintains an edge through to the final epoch.

<div align="center">
  <img src="benchmarks/results1/Vt1_acc_epoch.png" style="width:80%;" />
</div>

---

### Training and Validation Loss

Nami exhibits a smoother and steeper drop in training loss, indicating better convergence. Its validation loss remains lower than competitors in later epochs, showcasing improved generalization.

| Epoch | Train Loss (Nami) | Train Loss (Mish) | Train Loss (Swish) |
|-------|--------------------|-------------------|---------------------|
| 50    | **1.08**           | 1.11              | 1.28                |
| 100   | **0.66**           | 0.76              | 0.83                |

<div align="center">
  <img src="benchmarks/results1/T_loss_vs_epoch.png" style="width:80%;" />
</div>

<div align="center">
  <img src="benchmarks/results1/V_loss_vs_epoch.png" style="width:80%;" />
</div>

---

### Accuracy Progression

Nami achieves faster gains in early epochs while retaining higher final accuracy.

<div align="center">
  <img src="benchmarks/results1/Tt_acc_epoch.png" style="width:80%;" />
</div>

---

### Comparative Plot

<div align="center">
  <img src="benchmarks/results1/Nami_vs_others_plot.png" style="width:80%;" />
</div>

This consolidated plot highlights Nami’s consistent performance advantage across the entire training process.

---

## Key Observations

- **Stability**: Nami maintains stable training curves with fewer spikes in validation metrics.
- **Final Accuracy**: Achieves a new benchmark of **67.28%** on CIFAR-100 with ResNet-50.
- **Gradient Smoothness**: A favorable derivative curve helps optimize deep architectures effectively.

---

## Conclusion

Nami is a promising new activation function that achieves **state-of-the-art** performance on a challenging image classification benchmark. Its smooth gradient behavior and strong convergence properties make it an excellent choice for training deep models.

---

## Citation

If you use Nami in your research, please consider citing the repository or this report. For feedback, reproduction scripts, or extended benchmarks (e.g., ImageNet, ViTs), feel free to reach out or file an issue on the GitHub repository.
