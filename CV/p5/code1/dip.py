import numpy as np
import torch
import torch.nn as nn


class EncDec(nn.Module):
    def __init__(self):
        super(EncDec, self).__init__()
        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1) # Encoder - 1
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1) # Encoder - 2 
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1) # Encoder - 3 
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3_t = nn.Conv2d(64, 32, 3, 1, 1) # Decoder - 3
        self.conv2_t = nn.Conv2d(32, 16, 3, 1, 1) # Decoder - 2
        self.conv1_t = nn.Conv2d(16, 1, 3, 1, 1) # Decoder - 1
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn32((F.relu(self.conv2(out))))
        out = self.bn64(F.relu(self.conv3(out)))
        out = self.upsample(self.bn32(F.relu(self.conv3_t(out))))
        out = self.upsample(self.bn16(F.relu(self.conv2_t(out))))
        return self.upsample((F.relu(self.conv1_t(out))))