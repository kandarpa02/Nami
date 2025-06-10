import torch
import torch.nn as nn

@torch.jit.script
def nami(
       _x:torch.Tensor,
       w=0.3,
       a=1.0,
       b=1.5
):

    return torch.where(_x > 0, torch.tanh(_x * a) , torch.sin(_x * w)/b)