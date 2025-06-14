# Nami
Official Repository for Nami — an adaptive activation function with built-in strong regularization capabilities.

---

<img src="media/wave.jpeg" alt="Wave" width="90%">

*Nami means "wave" in Japanese — the name comes from its wavy nature in the negative domain.*
*Unlike typical activation functions that tend to a single value, Nami oscillates due to the sin* *function. It combines this oscillation with the smoothness of tanh. The behavior is controlled by* *three learnable parameters: `w`, `a`, and `b`.*

---

## Install

```bash
pip install git+https://github.com/kandarpa02/Nami.git

```

## Usage

```python
# for torch
from nami import Torch.Nami
nami_torch = Torch.Nami()

# for tensorflow
from nami import TF.Nami
nami_tf = TF.Nami()

# for jax
from nami import JAX.Nami
nami_jax = JAX.Nami()
```


## Compatible: 

```bash
PyTorch
TensorFlow
Jax (needs some fixes)

```

---

## Plots:

### Nami vs Others:

- Nami consistently leads in high-impact learning phases (e.g., around LR drops), emphasizing its adaptability.
- This resilience is not observed in traditional activations that either saturate early or destabilize under aggressive scheduling.

<img src="media/Nami_others_plot.png" alt="nami_vs_others" width="80%">

### Derivative of Nami:

- **Nami’s derivative exhibits a non-monotonic, softly saturating profile**—a design that diversifies gradient propagation and avoids neuron inactivation.
- Its flexible form enables localized feature modulation, especially beneficial in deep architectures where gradient dynamics are critical.

<img src="media/Nami_derivative.png" alt="derivative of nami" width="80%">

---


Here is a quick comparison of **Nami**, **Swish** and **Mish** with the same weight initialization on ResNet-18, on CIFAR-10, **More Detailed and complex tests are here** [benchmarks](benchmarks)

```python
def seed_everything(seed=42):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(42)
```

---

Ran the training for 200 epochs and `Nami` showed very stable losses troughout the training and especially in the later epochs and that is because of `tanh(x * a)` in the positive and `a * sin(x * w) / b` in the negative domain. It has learnable parameters `w`, `a`, `b`, which demand longer runs to learn deeper and complex information from the data.

- `w` is responsible for maintaining the wave-length, the smaller it is the smoother the 
    gradients are.

- `a` regulates the spikes of the waves, high waves can capture deeper information, but if it rises too much, it may cause overfitting — that’s where `b` comes into the picture.

- `b` tackles overfitting by supressing `a`'s dominance, and increases generalization.

<img src="media/nami_equation.png" alt="nami eq" width="75%">

---

Here is a sample benchmark on Resnet18:


<img src="media/Nami_activation_ResNet18.png" alt="Nami" width="60%">

**Nami**: 
Validation Accuracy `94.81`,
Training Loss `0.0015`,
Validation Loss `0.1963`

---

<img src="media/Mish_Resnet18.png" alt="Mish" width="60%">

**Mish**:
Validation Accuracy `94.09`,
Training Loss `0.0032`,
Validation Loss `0.2424`

---

<img src="media/Swish_resnet18.png" alt="Swish" width="60%">

**Swish/SiLU**:
Validation Accuracy `94.06`,
Training Loss `0.0024`,
Validation Loss `0.2347`

---

In conclusion, **Nami** beats **Mish** and **Swish** in both generalization and accuracy! **Nami** is suitable for **Larger** and **Deeper** Networks. And also; for **shallow Neural Networks** it captures more complex information than any other activation. Despite having three trainable parameters, its computational cost remains minimal — and the convergence is outstanding. For **more detailed** experiments visit [benchmarks](benchmarks).

---

"I'd love to see ML folks fine-tune LLMs or train deep models using **Nami**, you can share the stats with me here:
[Gmail](kandarpaexe@gmai.com), 
[X](x.com/_kandarpasarkar)

---

**Thanks for your support :)**