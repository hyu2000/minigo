""" PyTorch2 DNN interface

TF->Torch migration
https://www.gilsho.com/post/tensorflow_to_pytorch/
https://github.com/gilshm/mlperf-pytorch/blob/master/models/resnet.py
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

Conv2D = keras.layers.Conv2D
conv2d_kwargs = dict(padding='same', kernel_initializer='he_normal', data_format='channels_last')


def residual_module(layer_in, n_filters, kernel_size=(3, 3)):
    """
    this aligns more w/ AGZ resnet setup
    """
    merge_input = layer_in
    # check if the number of filters needs to be increase
    if layer_in.shape[-1] != n_filters:
        x = Conv2D(n_filters, (1, 1), activation='relu', **conv2d_kwargs)(layer_in)
        merge_input = keras.layers.BatchNormalization()(x)
    # conv1
    x = Conv2D(n_filters, kernel_size, activation=None, **conv2d_kwargs)(layer_in)
    x = keras.layers.BatchNormalization()(x)
    conv1 = keras.layers.Activation('relu')(x)
    # conv2
    x = Conv2D(n_filters, kernel_size, activation=None, **conv2d_kwargs)(conv1)
    conv2 = keras.layers.BatchNormalization()(x)
    # add
    x = keras.layers.add([conv2, merge_input])
    layer_out = keras.layers.Activation('relu')(x)
    return layer_out


class ResidualModule(nn.Module):
    """ GPT4 generated to migrate from residual_module """
    def __init__(self, in_channels, n_filters, kernel_size=3):
        super(ResidualModule, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, n_filters, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(n_filters)

        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(n_filters)

        self.increase_channels = in_channels != n_filters

    def forward(self, x):
        if self.increase_channels:
            merge_input = F.relu(self.bn1x1(self.conv1x1(x)))
        else:
            merge_input = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += merge_input
        return F.relu(out)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class ResBlock(nn.Module):
    def __init__(self, n_filters: int, kernel_size=(3, 3)):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(planes, eps=1.001e-5)
        self.bn2 = nn.BatchNorm2d(planes, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        # check if the number of filters needs to be increase

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



def build_model():
    """
    """