import torch
import torch.nn as nn

def nami(_x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    orig_dtype = _x.dtype
    _x = _x.to(torch.float32)

    w = torch.clamp(w, 0.1, 0.5).to(torch.float32)
    a = torch.clamp(a, 0.5, 3.0).to(torch.float32)
    b = torch.clamp(b, 0.5, 3.0).to(torch.float32)

    out = torch.where(_x > 0, torch.tanh(_x * a), a * torch.sin(_x * w) / b)
    return out.to(orig_dtype)
