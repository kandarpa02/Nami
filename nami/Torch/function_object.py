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
        self.w = w_init
        self.a = a_init
        self.b = b_init
        self.learnable = learnable

    def forward(self, x):
        nami(x, w_init=self.w, a_init=self.a, b_init=self.b, learnable=self.learnable)
