# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os

from collections import OrderedDict

class TransitionBlock(nn.Sequential):
    def __init__(self, num_input_feature, num_output_features):
        super(TransitionBlock, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))

class ConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class IntensiveLayer(nn.Module):

    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(IntensiveLayer, self).__init__()
        self.branch1x1 = ConvBnRelu(in_channels, growth_rate, kernel_size=1) # 1

        self.branch5x5_1 = ConvBnRelu(in_channels, bn_size*growth_rate, kernel_size=1)
        self.branch5x5_2 = ConvBnRelu(bn_size*growth_rate, growth_rate, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = ConvBnRelu(in_channels, bn_size*growth_rate, kernel_size=1)
        self.branch3x3dbl_2 = ConvBnRelu(bn_size*growth_rate, growth_rate, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = ConvBnRelu(growth_rate, growth_rate, kernel_size=3, padding=1)

        self.branch_pool = ConvBnRelu(in_channels, growth_rate, kernel_size=1)

        self.branch_1x1_output = ConvBnRelu(4 * growth_rate, growth_rate, kernel_size=1)

        self.drop_rate = drop_rate

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # Dropout
        outputs = torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 1)
        if self.drop_rate > 0:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)

        outputs = self.branch_1x1_output(outputs)
        return torch.cat([x, outputs], 1)

class IntensiveBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, drop_rate):
        super(IntensiveBlock, self).__init__()
        for i in range(num_layers):
            layer = IntensiveLayer(in_channels+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("intensive{}".format(i + 1), layer)

class Intensive224(nn.Sequential):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 32, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=1000):
        super(Intensive224, self).__init__()

        # First 2 layers of Conv2d
        self.features_d = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features // 2, kernel_size=7, stride=2, padding=3, bias=True)),
            ("norm0", nn.BatchNorm2d(num_init_features // 2)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(3, stride=2, padding=1)),
            ("conv1", nn.Conv2d(num_init_features // 2, num_init_features, kernel_size=3, stride=1, padding=1, bias=True)),
            ("norm1", nn.BatchNorm2d(num_init_features)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        self.features = nn.Sequential()
        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = IntensiveBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("intensiveblock{}".format(i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = TransitionBlock(num_features, int(num_features * compression_rate))
                self.features.add_module("transition{}".format(i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Sequential()
        self.classifier.add_module("fc1", nn.Linear(1024 * 4 * 4, out_features=128, bias=True))
        self.classifier.add_module("norm6", nn.BatchNorm1d(128))
        self.classifier.add_module("relu6", nn.ReLU(inplace=True))
        self.classifier.add_module("fc2", nn.Linear(128, 14, bias=True))

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features_d(x)
        features = self.features(features)
        out = F.avg_pool2d(features, 3, stride=2, padding=1)
        out = out.view(-1, self.num_flat_features(out))
        out = self.classifier(out)
        return out

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        for s in size:
            num_features *= s
        return num_features