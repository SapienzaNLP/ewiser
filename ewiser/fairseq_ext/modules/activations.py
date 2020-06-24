import torch
from torch.nn import functional as F

# https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/functional.py
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))

def penalized_tanh(input:torch.Tensor, alpha:float=0.25):
    penalization = (input < 0).to(dtype=input.dtype) * alpha
    return input.tanh() * penalization

def swish(input:torch.Tensor):
    return input * torch.sigmoid(input)
