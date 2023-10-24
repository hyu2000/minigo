""" PyTorch2 DNN interface

TF->Torch migration
https://www.gilsho.com/post/tensorflow_to_pytorch/
https://github.com/gilshm/mlperf-pytorch/blob/master/models/resnet.py
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
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


class ResBlock(nn.Module):
    """ GPT4 generated to migrate from residual_module

    to convert 'channels_last' to 'channels_first' (default in Torch):
    tensor = tensor.permute(0, 3, 1, 2)  # Convert (batch, height, width, channels) to (batch, channels, height, width)
    """
    def __init__(self, in_channels, n_filters, kernel_size=3):
        super(ResBlock, self).__init__()

        self.conv1x1 = self._conv2d(in_channels, n_filters, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(n_filters)

        self.conv1 = self._conv2d(in_channels, n_filters, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = self._conv2d(n_filters, n_filters, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(n_filters)

        self.increase_channels = in_channels != n_filters

    def _conv2d(self, in_channels, out_channels, kernel_size):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        init.kaiming_normal_(conv.weight, nonlinearity='relu')
        return conv

    def forward(self, x):
        if self.increase_channels:
            merge_input = F.relu(self.bn1x1(self.conv1x1(x)))
        else:
            merge_input = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += merge_input
        return F.relu(out)


class CustomModel(nn.Module):
    def __init__(self, input_shape):
        super(CustomModel, self).__init__()

        # Common features
        self.pad = nn.ConstantPad2d((0, 0, 0, 0, 0, 1), 1)

        # Value head
        self.conv_value = self._conv2d(input_shape[2] + 1, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(input_shape[0] * input_shape[1], 64)
        self.fc_value2 = nn.Linear(64, 1)

        # Policy head
        self.conv_policy = self._conv2d(input_shape[2] + 1, 1, kernel_size=1)
        self.fc_policy = nn.Linear(input_shape[0] * input_shape[1] + 4, 82)

    def _conv2d(self, in_channels, out_channels, kernel_size):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        return conv

    def forward(self, x):
        x = self.pad(x)

        # todo do we need multiple ResBlock()?
        x = residual_module(x, 32, (5, 5))
        for i in range(5):
            x = residual_module(x, 64, (3, 3))

        features_common = x

        # Value head
        x_value = F.relu(self.bn_value(self.conv_value(features_common)))
        x_value = x_value.view(x_value.size(0), -1)
        x_value = F.relu(self.fc_value1(x_value))
        output_value = torch.tanh(self.fc_value2(x_value))

        # Policy head
        x_policy = self.conv_policy(features_common)
        move_prob = x_policy.view(x_policy.size(0), -1)
        pass_inputs = torch.stack([
            torch.mean(move_prob, dim=1),
            torch.max(move_prob, dim=1),
            torch.std(move_prob, dim=1),
            output_value.squeeze(dim=1)
        ], dim=1)
        pass_prob = self.fc_policy(pass_inputs)
        x_policy = torch.cat([move_prob, pass_prob], dim=1)
        output_policy = F.softmax(x_policy, dim=1)

        return output_policy, output_value


# Example usage:
# model = CustomModel((8, 8, 3))
