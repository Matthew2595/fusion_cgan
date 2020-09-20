# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Plot three images in line for optical image visualization.
Draw also rectangles of given size.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

from matplotlib import rcParams
rcParams['axes.titlesize'] = 90
rcParams['axes.labelsize'] = 90

folder = input('Gimme the folder:')
rows = 1
cols = 3
img='o0.jpg'
sizes = [(10980,9280),(10980,10980),(10980,10240)]

def display_multiple_img(images, rows, cols):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    figure.set_figheight(15)
    figure.set_figwidth(55)
    figure.set_dpi(100)
    figure.subplots_adjust(hspace=1)
    figure.subplots_adjust(wspace=1)
    for ind,key in enumerate(images):
        ax.ravel()[ind].imshow(Image.open(images[key], mode='r'))
        if ind==0:
            ax.ravel()[ind].set_title('North')
        elif ind==1:
            ax.ravel()[ind].set_title('Centre')
        if ind==2:
            ax.ravel()[ind].set_title('South')

        ax.ravel()[ind].add_patch(Rectangle((10,10), 3*sizes[ind][0]/4-40, sizes[ind][1]-15, facecolor='None', edgecolor="red", linewidth=10))
        ax.ravel()[ind].add_patch(Rectangle((3*sizes[ind][0]/4+30,10), sizes[ind][0]/4-50, sizes[ind][1]-15, facecolor='None', edgecolor="yellow", linewidth=10))

        ax.ravel()[ind].set_axis_off()
        print('img check')
    plt.tight_layout()
    plt.show()

images = {'Image0': os.path.join(folder, '0_North', '0', img)
          , 'Image1': os.path.join(folder, '1_Centre', '0', img)
          , 'Image2': os.path.join(folder, '2_South', '0', img)}


display_multiple_img(images, rows, cols)
