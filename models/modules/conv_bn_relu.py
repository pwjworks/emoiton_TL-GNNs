# -*- coding: utf-8 -*-
"""

"""
from torch.nn import Conv2d, BatchNorm2d, Sequential, ReLU


def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels,
               kernel_size=kernel_size, stride=stride, padding=padding),
        BatchNorm2d(out_channels),
        ReLU()
    )
