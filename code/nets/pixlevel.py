# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

'''pixel-level module'''


class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2 
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),  # 2, 2*self.
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    '''forward'''

    def forward(self, x):
        x_avg = self.conv_avg(x)  # [b, 512, 28, 28]
        x_avg = self.relu_avg(x_avg)  # [b, 512, 28, 28]
        x_avg = torch.mean(x_avg, dim=1)  # [b, 28, 28]
        x_avg = x_avg.unsqueeze(dim=1)  # [b, 1, 28, 28]
        x_max = self.conv_max(x)  # [b, 512, 28, 28]
        x_max = self.relu_max(x_max)  # [b, 512, 28, 28]
        x_max = torch.max(x_max, dim=1).values  # [b, 28, 28]
        x_max = x_max.unsqueeze(dim=1)  # [b, 1, 28, 28]
        x_out = x_max+x_avg  # [b, 1, 28, 28]
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)  # [b, 3, 28, 28]
        x_output = x_output.transpose(1, 3)  # [b, 28, 28, 3]
        x_output = self.bottleneck(x_output)  # [b, 28, 28, 1]
        x_output = x_output.transpose(1, 3)  # [b, 1, 28, 28]
        y = x_output * x  # [b, 512, 28, 28]
        return y  # [b, 512, 28, 28]
