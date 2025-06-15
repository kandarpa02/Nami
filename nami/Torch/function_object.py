from nami.Torch.functional import nami

try:
    import torch.nn as nn
    import torch
except:
    raise ImportError("PyTorch is not installed")

        
class Nami(nn.Module):
    '''
    PyTorch implementation of Nami

    Nami means wave in Japanese, the name came from its wavy nature in the negative domain
    due to the `sin` function, rather than tending to one value like other functions
    `Nami` oscillates in the negative side, and has the smoothness of `tanh`. According to
    the training data the oscilation is maintained by three learnable parameters: `w`, `a`, `b`.

    Parameters:
        w: Controls wavelength of sin (smoothing)
        a: Controls amplitude of wave
        b: Regulates overfitting by suppressing a
        learnable: Whether parameters are trainable
    

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
        return nami(_x=x, w=self.w, a=self.a, b=self.b)
