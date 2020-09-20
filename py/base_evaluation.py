# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Basic loop to get baseline value for performance comparison between simulated
images and objective ones.
It just takes the two optical images as source and gets the indices for them.
"""

from config import *
from metrics import PSNR, SSIM

from PIL import Image
import os
import torch
import torchvision.transforms as transforms

Image.MAX_IMAGE_PIXELS = 1000000000

psnr = PSNR()
ssim = SSIM()

for region in regions:
    o0 = Image.open(os.path.join(source_folder, region, 'o0.jpg'))
    o0 = transforms.ToTensor().__call__(o0)
    o0 = o0.unsqueeze(0)
    o1 = Image.open(os.path.join(source_folder, region, 'o1.jpg'))
    o1 = transforms.ToTensor().__call__(o1)
    o1 = o1.unsqueeze(0)

    psnr_value = psnr(o1, o0).item()
    ssim_value = ssim(o1, o0).item()
    
    print('>>>INDICES FOR %s<<<' % (region[2:]))
    print('PSNR: %.4f SSIM: %.4f' % (psnr_value, ssim_value))

