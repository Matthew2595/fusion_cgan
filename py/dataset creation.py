# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

This script aims to create a dataset of random patches of the original image to
train and validate the network. The image is divided into some parts, these ones
are saved to the data folder and the main code will call them to create the dataset
using the right module of torchvision.
"""

from config import *

"""Built using the Pillow library, so images are read as images, intead of plots."""
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import random
import os


'''FUNCTIONS'''


def make_dir(root):
    """
    This function wants to create the correct tree of directories. In this way,
    the following lines of code will work correctly in a standardized environment.

    The directory are as follows:
    root/train0/images
    root/train1/images
    ...

    Args:
        root (str): Root directory poth.
    """

    try:
        for s in switch:
            os.mkdir(os.path.join(root, s))
            for r in regions:
                os.mkdir(os.path.join(root, s, r))
                for i in range(train_samples):
                    if s=='validation' and i==eval_samples:
                        break
                    os.mkdir(os.path.join(root, s, r, s+str(i)))
    except FileExistsError:
        pass
        #print('Remove all the folders in dataset and assure it is empty before running this script.')


def check_size(imgs_dict, region):
    """
    This function check that all the pictures loaded for a region are the same.
    It takes the dictionary with the pictures and the region we are processing.
    It creates a list to save the sizes of all the picture and check if these
    values are equal.

    Args:
        imgs_dict (dict): dictionary with images (i.e. img_names).
        region (str): region it is processing.
    """

    check_list = []
    for i, name in enumerate(imgs_dict):
        check_list.insert(i, imgs_dict[name].size)
    result = check_list.count(check_list[0]) == len(check_list)
    if result:
        print('For ' + region + ' sizes checked, everything good')
    else:
        raise ValueError('Sizes of images are not equal!')


def crop_and_save(folder, imgs_dict, cropped_dict, area, id_sample, switch, region):
    """
    Cycle to crop the main images, save them in another dictionary to not change
    the original one, and save them in the approriate folder according to the
    cycle.

    Args:
        folder (str): is the path for dataroot folder.
        imgs_dict (dict): is dictionary with the main images.
        cropped_dict (dict): dictionary created at the beginning with same keys
            as imgs_dict, but used to save cropped images.
        area (tuple): stores the coordinates to crop the images.
        id_sample (int): is the sequential identifier for the sample.
        switch (str): split between training and validation.
        region (str): region it is processing.
    """

    for name in cropped_dict:
        cropped_dict[name] = imgs_dict[name].crop(area)
        cropped_dict[name].save(os.path.join(folder, switch, region, switch+str(id_sample), name))


def create_dataset(root, source):
    """
    This function creates the dataset of little images to feed the network.
    A randome couple of coordinates is created. From these coordinates an area
    is created as a tuple. This values are used to crop the originale image.
    The cropped image is saved into a backup list and then a new .jpg file is
    written with a progressive number.
    The difference between training and validating data is made constraining
    the crop operation to two different areas.

    Args:
        root (str): root directory path (i.e. data_folder).
        source (str): folder where to fetch main images (i.e. source_folder).

    result (list): list of two numbers, they are the coordinates from which
        start the crop.
    """

    result = [0, 0]
    for r in regions:
        print('^^^^^^^')
        print("I'm loading the region " + r)

        # The images are read and opened in the dictionary
        for name in imgs:
            imgs[name] = Image.open(os.path.join(source, r, name))
            # Taking advantage from the dict, the size of pictures is saved
            width, height = imgs[name].size

        # Function to check that the pictures have the same size
        check_size(imgs, r)

        # The limit for random functions is set
        # Train and valid data are split in width (25% of it)
        height_limit = height - imgs_size
        width_limit = width - imgs_size
        split_value = width - round(width/4)

        for s in switch:
            for i in range(train_samples):
                # The area is created according to the type of value we need
                if s == 'train':
                    result[0] = random.randint(0, split_value - imgs_size)
                if s == 'validation':
                    if i == eval_samples:
                        break
                    result[0] = random.randint(split_value, width_limit)
                result[1] = random.randint(0, height_limit)
                area = (result[0], result[1], result[0]+imgs_size, result[1]+imgs_size)

                # Images are cropped, like if they were one over another
                crop_and_save(root, imgs, cropped_imgs, area, i, s, r)

            print('Cropped and saved region:', r, 'for', s)


'''SCRIPT TO RUN'''

make_dir(data_folder)
create_dataset(data_folder, source_folder)
