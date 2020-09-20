# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Here there's the code for the generator and discriminator nets.
There are the functions for convolutional and ResNext blocks and then the
nets classes.
"""

import torch
import torch.nn as nn
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def ConvBlock(in_channels, out_channels, kernel, stride, padding, leaky):
    """
    In this function we define the standard convolutional block to use in the
    generator and discriminator nets: it is composed of a convolutional layer,
    Batch normalization and ReLu activation function (that becomes LeakyReLu in
    the discriminator thanks to a boolean value in the networks). The forward 
    function is implemented inside the network.

    Args:
        in_channels (int): input channels number.
        out_channels (int): output channels number.
        kernel (int): size of the kernel (square kernel only available in this case).
        stride (int): stride value.
        padding (int): padding value.

    Returns:
        object: the layers of the convolutional block made with Seuqential
            container.
    """

    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding))
    layers.append(nn.BatchNorm2d(out_channels))
    if leaky:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    else: layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def ResNext(channels, cardinality, probability):
    """
    This is the function for the ResNeXt block to use in the generator net. It is
    an evolution of ResNet framework: it is composed of a convolutional layer, Batch
    normalization, ReLu activation function, Dropout layer and then the first three
    layer repeated. It is worth saying that an another activation function (ReLU) is 
    needed to complete the block and it's applied in the forward function of the net 
    on the sum of the block's output and the residual value. The different paths are
    created via grouped convolution.

    Args:
        channels (int): number of channels to pass through the layers; only one
            value is needed since it does not change.
        cardinality (int): number used to split the computation of the conv
            layers in different groups, or paths.
        probability (float): it is the probability used for the dropout.

    Returns:
        object: the layers of the resnext block made with Seuqential container.
    """

    layers = []
    layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=cardinality))
    layers.append(nn.BatchNorm2d(channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Dropout2d(p=probability))
    layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=cardinality))
    layers.append(nn.BatchNorm2d(channels))
    return nn.Sequential(*layers)


class Gen(nn.Module):
    """
    This module is for the Generator net. Given an input as a tensor, it generate
    an image of the size needed. Since the GAN is conditional, it takes a series
    of images plus a random noise picture. The prediction is a fake image that aims to
    resemble the real image. It is composed of three convolutional blocks (previously defined),
    several ResNext blocks (already defined), two other convolutional blocks and, after a
    convolutional operation, a Tanh activation function is applied. In this case,
    the convolutions are applied to maintain the same input_size throughout the net,
    while the number of features change.

    Args:
        input_feat (int): input features, or rathe channels.
        features (list): it stores the number of features of each block as ints;
            with this setting it needs 6 values.
        ngpu (int): number of gpus available.
        probability (float): value in range [0,1] used to apply dropout.

    Attributes:
        leaky (boolean): it is used to create a conv block with LeakyReLu function,
            set False in the generator.
        block (function): convolutional blocks as defined by the function.
        res (function): resnext blocks as defined by the function.
        relu (function): relu activation function for resnext.
        conv (function): last convolutional layer.
        tanh (function): hyperbolic tan function as last activation.
    """

    def __init__(self, input_feat, features, ngpu, probability):
        super(Gen, self).__init__()
        self.ngpu = ngpu
        self.leaky = False
        self.block0 = ConvBlock(input_feat, features[0], 7, 1, 3, self.leaky)
        self.block1 = ConvBlock(features[0], features[1], 3, 1, 1, self.leaky)
        self.block3 = ConvBlock(features[1], features[2], 3, 1, 1, self.leaky)
        self.res1 = ResNext(features[2], 32, probability)
        self.relu1 = nn.ReLU()
        self.res2 = ResNext(features[2], 32, probability)
        self.relu2 = nn.ReLU()
        self.res3 = ResNext(features[2], 32, probability)
        self.relu3 = nn.ReLU()
        self.block4 = ConvBlock(features[2], features[3], 3, 1, 1, self.leaky)
        self.block5 = ConvBlock(features[3], features[4], 3, 1, 1, self.leaky)
        self.conv = nn.Conv2d(features[4], features[5], kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        It is the function that defines the calculation made throughout the net.
        """

        y = self.block0(x)
        y = self.block1(y)
        y = self.block3(y)
        ry = self.res1(y)
        y = self.relu1(y + ry)
        ry = self.res2(y)
        y = self.relu2(y + ry)
        ry = self.res3(y)
        y = self.relu3(y + ry)
        y = self.block4(y)
        y = self.block5(y)
        y = self.conv(y)
        return self.tanh(y)


class Dis(nn.Module):
    """
    This module is for the Discriminator net. It takes an image as input and
    return a 2d tensor with zeros or ones: this "patch" is the label of the input
    image, i.e. scalar probability that the input image is real (as opposed to fake).
    Both the real and fake images, in turn, are processed and the output is used
    to evaluate the loss and training the net. It is composed of a series of convolutional
    blocks with convolution, batch normalization and LeakyReLU, and a sigmoid function
    at the end to squeeze the output between [0,1]. The sigmoid function was deleted
    as layer and it was added embedded in the BCEloss with logits. The result is the same.

    Args:
        input_feat (int): input features, or rathe channels.
        features (list): it stores the number of features of each block as ints;
            with this setting it needs 5 values.
        ngpu (int): number of gpus available.

    Attributes:
        leaky (boolean): it is used to create a conv block with LeakyReLu function,
            set True in the generator.
        block (function): convolutional blocks as defined by the function.
        res (function): resnext blocks as defined by the function.
        relu (function): leaky relu activation function.
        conv (function): convolutional layers.
        tanh (function): hyperbolic tan function as last activation.
    """

    def __init__(self, input_feat, features, ngpu):
        super(Dis, self).__init__()
        self.ngpu = ngpu
        self.leaky = True
        self.conv0 = nn.Conv2d(input_feat, features[0], kernel_size=4, stride=2, padding=1)
        self.relu0 = nn.LeakyReLU()
        self.block1 = ConvBlock(features[0], features[1], 4, 2, 1, self.leaky)
        self.block2 = ConvBlock(features[1], features[2], 4, 2, 1, self.leaky)
        self.block3 = ConvBlock(features[2], features[3], 4, 1, 1, self.leaky)
        self.conv1 = nn.Conv2d(features[3], features[4], kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        """
        It is the function that defines the calculation made throughout the net.
        """

        y = self.conv0(x)
        y = self.relu0(y)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.conv1(y)
        return y
