# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Script used to plot the 12 images as final results in a grid with correct labels.
"""

import matplotlib.pyplot as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

from matplotlib import rcParams
rcParams['axes.titlesize'] = 90
rcParams['axes.labelsize'] = 90


folder = input('Gimme the folder: ')
rows = 4
cols = 3

def display_multiple_img(images, rows, cols):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    figure.set_figheight(50)
    figure.set_figwidth(55)
    figure.set_dpi(100)
    figure.subplots_adjust(hspace=0.5)
    figure.subplots_adjust(wspace=0.2)
    for ind,key in enumerate(images):
        ax.ravel()[ind].imshow(Image.open(images[key], mode='r'))
        if ind==0:
            ax.ravel()[ind].set_title('North')
            ax.ravel()[ind].set_ylabel('Test 1 models')
        elif ind==1:
            ax.ravel()[ind].set_title('Centre')
        if ind==2:
            ax.ravel()[ind].set_title('South')
        elif ind==3:
            ax.ravel()[ind].set_ylabel('Transfer 1')
        elif ind==6:
            ax.ravel()[ind].set_ylabel('Transfer 2')
        elif ind==9:
            ax.ravel()[ind].set_ylabel('Objective')
        if ind%3==0:
            ax.ravel()[ind].set_yticks([])
            ax.ravel()[ind].set_xticks([])
        else: ax.ravel()[ind].set_axis_off()
        print('img check')
    plt.tight_layout()
    plt.show()

images = {'Image0': os.path.join(folder, 'North', 'from N128','fake.jpg')
          , 'Image1': os.path.join(folder, 'Centre', 'from C128','fake.jpg')
          , 'Image2': os.path.join(folder, 'South', 'from S128','fake.jpg')
          , 'Image3': os.path.join(folder, 'North', 'from transfer1','fake.jpg')
          , 'Image4': os.path.join(folder, 'Centre', 'from transfer1','fake.jpg')
          , 'Image5': os.path.join(folder, 'South', 'from transfer1','fake.jpg')
          , 'Image6': os.path.join(folder, 'North', 'from transfer2','fake.jpg')
          , 'Image7': os.path.join(folder, 'Centre', 'from transfer2','fake.jpg')
          , 'Image8': os.path.join(folder, 'South', 'from transfer2','fake.jpg')
          , 'Image9': os.path.join(folder, 'North', 'from transfer2','real.jpg')
          , 'Image10': os.path.join(folder, 'Centre', 'from transfer2','real.jpg')
          , 'Image11': os.path.join(folder, 'South', 'from transfer2','real.jpg')}

display_multiple_img(images, rows, cols)

