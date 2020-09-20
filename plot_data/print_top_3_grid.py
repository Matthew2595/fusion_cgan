# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

In combination with top_3 script, this one plot the top 3 patches with their values.
"""

from get_top_3 import *

import matplotlib.pyplot as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

from matplotlib import rcParams
rcParams['axes.titlesize'] = 35
rcParams['font.size'] = 40

# from the other file
#folder = input('gimme the folder: ')
region = input('Gimme the region: ')
rows = 2
cols = 3

def display_multiple_img(images, rows, cols):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    figure.set_figheight(15)
    figure.set_figwidth(20)
    figure.set_dpi(300)
    figure.subplots_adjust(hspace=0.2)
    figure.subplots_adjust(wspace=0.4)
    for ind,key in enumerate(images):
        ax.ravel()[ind].imshow(Image.open(images[key], mode='r'))
        ax.ravel()[ind].set_axis_off()
    plt.figtext(0.128, 0.5, ssim_1, va='center')
    plt.figtext(0.5, 0.5, ssim_2, va='center', ha='center')
    plt.figtext(0.775, 0.5, ssim_3, va='center')
    plt.figtext(-0.02, 0.5, region, va='center', ha="left", rotation=90, fontweight='bold')
#    plt.figtext(0.5, 0.98, 'SSIM values', ha="center")
    figure.suptitle('SSIM values', fontsize=40, fontweight='bold')
    plt.tight_layout()
    plt.show()

images = {'Image0': os.path.join(folder, 'validation', 'fake','save'+str(ssim_ind_1)+'.jpg')
          , 'Image1': os.path.join(folder, 'validation', 'fake','save'+str(ssim_ind_2)+'.jpg')
          , 'Image2': os.path.join(folder, 'validation', 'fake','save'+str(ssim_ind_3)+'.jpg')
          , 'Image3': os.path.join(folder, 'validation', 'real','save'+str(ssim_ind_1)+'.jpg')
          , 'Image4': os.path.join(folder, 'validation', 'real','save'+str(ssim_ind_2)+'.jpg')
          , 'Image5': os.path.join(folder, 'validation', 'real','save'+str(ssim_ind_3)+'.jpg')}

display_multiple_img(images, rows, cols)

