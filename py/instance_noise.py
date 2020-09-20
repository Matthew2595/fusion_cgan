# -*- coding: utf-8 -*-
"""
Spyder Editor

Script to add noise to images.
"""

from PIL import Image
import torchvision.transforms as transforms
import torch
#torch.set_default_dtype(torch.float64)
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb


def instance_noise(fake_image, real_image, iters, device='cpu'):
    '''
    This function is used  to test the effectiveness of instance noise in cGAN
    training. The idea comes from:
        https://arxiv.org/abs/1610.04490
        https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    The idea is to add AWGN to both real and fake samples before feeding the
    discriminator to make its job harder and stabilise the overall training.
    The tensor is prepared and noise is created. The noise variance decreases
    linealry according to the number of iterations we want to add the noise.
    Both tensors are converted to ndarrays, the image is converted to HSV values,
    noise is adde to the value layer, the image is brought back to RGB and
    returned as 4D tensor [N,C,H,W] on the requested device.

    Args:
        image (tensor): input image as a tensor, 3d is needed, but if it is more
            it is prepared for addition
        iters (int): number of current iteration, used to anneal the noise variation
        device (str): device used to store the resulting object to continue
            during training; set to cpu by default

    Return:
        tensor: 4d tensor with added noise on the given device ready to feed
            the discriminator
    '''

    global sigma
    global max_iters

    # Tensor prepared to be in 3D
    if len(fake_image.shape)==4:
        fake_image = fake_image.squeeze(0)
    if len(real_image.shape)==4:
        real_image = real_image.squeeze(0)

    # Function exit is noise is no longer needed
    if iters==max_iters:
        return fake_image.unsqueeze(0).to(device), real_image.unsqueeze(0).to(device)

    # Noise is created and converted to ndarray
    sizes = fake_image.size()
    add_noise = (sigma - iters * (sigma/max_iters)) * torch.randn(sizes[1], sizes[2])
    add_noise = add_noise.numpy()

    ### Fake and real images takes the same noise in different times ###
    # Fake image noise
    # Tensor image to ndarray and [C,H,W] >>> [H,W,C]
    np_img = fake_image.numpy()
    np_img = np.transpose(np_img, (1,2,0))

    hsv_img = rgb2hsv(np_img)
    hsv_img[:, :, 2] = hsv_img[:, :, 2] + add_noise
    rgb_img = hsv2rgb(hsv_img)

    # ndarray to tensor and [H,W,C] >>> [C,H,W]
    rgb_img = np.transpose(rgb_img, (2,0,1))
    fake_image = torch.from_numpy(rgb_img)

    # Real image noise
    np_img = real_image.numpy()
    np_img = np.transpose(np_img, (1,2,0))

    hsv_img = rgb2hsv(np_img)
    hsv_img[:, :, 2] = hsv_img[:, :, 2] + add_noise
    rgb_img = hsv2rgb(hsv_img)

    # ndarray to tensor and [H,W,C] >>> [C,H,W]
    rgb_img = np.transpose(rgb_img, (2,0,1))
    real_image = torch.from_numpy(rgb_img)

    # 4D tensor on device
    return fake_image.unsqueeze(0).to(device), real_image.unsqueeze(0).to(device)

