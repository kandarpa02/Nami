import torch
import torch.nn as nn

@torch.jit.script
def nami(
       _x:torch.Tensor,
       w:torch.Tensor,
       a:torch.Tensor,
       b:torch.Tensor
):
    return torch.where(_x > 0, torch.tanh(_x * a) , a * torch.sin(_x * w)/b)