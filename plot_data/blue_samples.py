# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Used to print a bunch of different samples.
"""


import matplotlib.pyplot as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

from matplotlib import rcParams
rcParams['axes.titlesize'] = 90
rcParams['axes.labelsize'] = 90


folder = input('Gimme the folder:')
rows = 1
cols = 4

def display_multiple_img(images, rows, cols):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    figure.set_figheight(5)
    figure.set_figwidth(20)
    figure.set_dpi(300)
    figure.subplots_adjust(hspace=1)
    figure.subplots_adjust(wspace=1)
    for ind,key in enumerate(images):
        ax.ravel()[ind].imshow(Image.open(images[key], mode='r'))
        ax.ravel()[ind].set_axis_off()
        print('img check')
    plt.tight_layout()
    plt.show()

images = {'Image0': os.path.join(folder, 'fake', 'save24120.jpg')
          , 'Image1': os.path.join(folder, 'fake', 'save24070.jpg')
          , 'Image2': os.path.join(folder, 'fake', 'save24520.jpg')
          , 'Image3': os.path.join(folder, 'fake', 'save24640.jpg')}


display_multiple_img(images, rows, cols)