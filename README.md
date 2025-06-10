# Nami
---

<img src="media/wave.jpeg" alt="Wave" width="90%">

*Nami means wave in Japanese, the name came from its wavy nature in the negative domain*
*due to the `sin` function, rather than tending to one value like other functions*
*`Nami` oscillates in the negative side, and has the smoothness of `tanh`. According to*
*the training data the oscilation is maintained by three learnable parameters: `w`, `a`, `b`.*

---

I tested **Nami**, **Swish** and **Mish** with the same weight initialization on ResNet-18, on CIFAR-10

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

Ran the training for 200 epochs and `Nami` showed very stable looses troughout the training and especially in the later epochs and that is because of `tanh(x * a)` in the positive and `a * sin(x * w) / b` in the negative domain. it has learnable parameters `w`, `a`, `b`, which demand longer runs to learn deeper and complex information from the data.

- `w` is responsible for maintaining the wave-length, the smaller it is the smoother the 
    gradients are.

- `a` regulates the spikes of the waves, high waves can capture deeper information, but if it
    keeps rising it will cause overfitting, then `b` comes into the picture.

- `b` tackles overfitting by supressing `a`'s dominance, and increses generalization.

<img src="media/nami_equation.png" alt="nami eq" width="75%">

---

And here are the stats:


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

If I conclude, **Nami** beats **Mish** and **Swish** in both generalization and accuracy! **Nami** is suitable for **Larger** and **Deeper** Networks. And also; for **shallow Neural Networks** it captures more complex information than any other activation. The computation cost is minimal for the fact that it has three trainable parameters, in return it shows outstanding convergence.

---

I highly appreciate if the ML folks Fine-tune LLMs or train Deep Learning models using **Nami**, you can share the stats with me here:
[Gmail](kandarpaexe@gmai.com)
[X link](x.com/_kandarpasarkar)

---

**Thanks for your support :)**