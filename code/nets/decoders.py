# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch.nn as nn
import torch
from .modules import ResConv


class GDecoder(nn.Module):
    def __init__(self, C=512, kernel_size=3, norm='ln', act='gelu', double=True, n_layers=2, **kwargs):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv(
                    C, C,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    upsample=True,
                    norm=norm,
                    activ=act,
                    double=double,
                    gate=True
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
