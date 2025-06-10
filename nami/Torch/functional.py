import torch
import torch.nn as nn

@torch.jit.script
def nami(
       _x:torch.tensor,
       w_init=0.3,
       a_init=1.0,
       b_init=1.5,
       learnable = True
):
    '''
    Nami means wave in japanese, the name came from its wavy nature in the negative domain
    due to the `sin` function, rather than tending to one value like other functions
    `Nami` oscillates in the negative side, and has the smoothness of `tanh`. According to
    the training data the oscilation is maintained by three learnable parameters: `w`, `a`, `b`.

    `w` is responsible for maintaining the wave-length, the smaller it is the smoother the 
        gradients are.

    `a` regulates the spikes of the waves, high waves can capture deeper information, but if it
        keeps rising it will cause overfitting, then `b` comes into the picture.
    
    `b` tackles overfitting by supressing `a`'s dominance, and increses generalization.

    '''
    
    if learnable:
            w = nn.Parameter(torch.tensor(w_init))
            a = nn.Parameter(torch.tensor(a_init))
            b = nn.Parameter(torch.tensor(b_init))
    else:
        w = torch.tensor(w_init)
        a = torch.tensor(a_init)
        b = torch.tensor(b_init)

    w = torch.clamp(w, min=0.1, max=0.5)
    b = torch.clamp(b, min=0.5, max=3.0)
    a = torch.clamp(a, min=0.5, max=3.0)

    return torch.where(_x > 0, torch.tanh(_x * a) , torch.sin(_x * w)/b)