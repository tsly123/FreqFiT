'''
Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers
https://arxiv.org/pdf/2111.13587.pdf
'''
import math
import torch
import torch.fft
import torch.nn as nn

class GlobalFilter(nn.Module):
    '''
    https://github.com/NVlabs/AFNO-transformer/blob/master/afno/gfn.py
    '''
    def __init__(self, blocks, dim, h=14):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(blocks, h, dim, 2, dtype=torch.float32) * 0.02)
        self.h = h

    def forward(self, block, x, dim=1):

        B, a, C = x.shape
        x = x.to(torch.float32)
        res = x
        x = torch.fft.rfft(x, dim=dim, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight[block].squeeze())
        x = x * weight
        x = torch.fft.irfft(x, n=a, dim=dim, norm='ortho')
        x = x + res

        return x