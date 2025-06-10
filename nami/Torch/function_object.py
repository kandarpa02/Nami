import torch.nn as nn
import torch
from nami.Torch.functional import nami

class Nami(nn.Module):
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

    def __init__(self, w_init=0.3, a_init = 1.0, b_init = 1.5, learnable=True):
        super().__init__()
        self.learnable = learnable
        if self.learnable:
            self.w = nn.Parameter(torch.tensor(w_init))
            self.a = nn.Parameter(torch.tensor(a_init))
            self.b = nn.Parameter(torch.tensor(b_init))
        else:
            self.w = torch.tensor(w_init)
            self.a = torch.tensor(a_init)
            self.b = torch.tensor(b_init)


    def forward(self, x):

        w = torch.clamp(self.w, min=0.1, max=0.5)
        a = torch.clamp(self.a, min=0.5, max=3.0)
        b = torch.clamp(self.b, min=0.5, max=3.0)

        return nami(_x=x, w=self.w, a=self.a, b=self.b)
