# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

This script plot all the scource images in one comprehensive grid.
"""

import matplotlib.pyplot as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

from matplotlib import rcParams
rcParams['axes.titlesize'] = 90
rcParams['axes.labelsize'] = 90


folder = input('Gimme the folder: ')
rows = 3
cols = 6

def display_multiple_img(images, rows, cols):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    figure.set_figheight(50)
    figure.set_figwidth(70)
    figure.set_dpi(90)
    figure.subplots_adjust(hspace=0.5)
    figure.subplots_adjust(wspace=0.1)
    for ind,key in enumerate(images):
        ax.ravel()[ind].imshow(Image.open(images[key], mode='r'))
        if ind==0:
            ax.ravel()[ind].set_title('Optical at t0')
            ax.ravel()[ind].set_ylabel('North')
        elif ind==1:
            ax.ravel()[ind].set_title('SAR VH at t0')
        if ind==2:
            ax.ravel()[ind].set_title('SAR VV at t0')
        elif ind==3:
            ax.ravel()[ind].set_title('Optical at t1')
        elif ind==4:
            ax.ravel()[ind].set_title('SAR VH at t1')
        elif ind==5:
            ax.ravel()[ind].set_title('SAR VV at t1')
        elif ind==6:
            ax.ravel()[ind].set_ylabel('Centre')
        elif ind==12:
            ax.ravel()[ind].set_ylabel('South')
        if ind%6==0:
            ax.ravel()[ind].set_yticks([])
            ax.ravel()[ind].set_xticks([])
        else: ax.ravel()[ind].set_axis_off()
        print('img check')
    plt.tight_layout()
    plt.show()

images = {'Image0': os.path.join(folder, '0_North', '0','o0.jpg')
          , 'Image1': os.path.join(folder, '0_North', '0','s0VH.jpg')
          , 'Image2': os.path.join(folder, '0_North', '0','s0VV.jpg')
          , 'Image3': os.path.join(folder, '0_North', '0','o1.jpg')
          , 'Image4': os.path.join(folder, '0_North', '0','s1VH.jpg')
          , 'Image5': os.path.join(folder, '0_North', '0','s1VV.jpg')
          , 'Image6': os.path.join(folder, '1_Centre', '0','o0.jpg')
          , 'Image7': os.path.join(folder, '1_Centre', '0','s0VH.jpg')
          , 'Image8': os.path.join(folder, '1_Centre', '0','s0VV.jpg')
          , 'Image9': os.path.join(folder, '1_Centre', '0','o1.jpg')
          , 'Image10': os.path.join(folder, '1_Centre', '0','s1VH.jpg')
          , 'Image11': os.path.join(folder, '1_Centre', '0','s1VV.jpg')
          , 'Image12': os.path.join(folder, '2_South', '0','o0.jpg')
          , 'Image13': os.path.join(folder, '2_South', '0','s0VH.jpg')
          , 'Image14': os.path.join(folder, '2_South', '0','s0VV.jpg')
          , 'Image15': os.path.join(folder, '2_South', '0','o1.jpg')
          , 'Image16': os.path.join(folder, '2_South', '0','s1VH.jpg')
          , 'Image17': os.path.join(folder, '2_South', '0','s1VV.jpg')}

display_multiple_img(images, rows, cols)