import numpy as np
import torch
import torch.nn as nn


class EncDec(nn.Module):

    def __init__(self):
        super(EncDec, self).__init__()
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

