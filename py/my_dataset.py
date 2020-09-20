# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Custom class to create a dataset for my need.
It subclasses VisionDataset, but it is programmed similar to DatasetFolder, from
which it takes the main idea.

In this case the sample is not the single image, but different images that
represent a single sample when merged together creating a single tensor
where the data si stored in different layers, or channels. So the single tensor
is our sample.
In this way we have data integrity, since the dataloader can shuffle keeping
all the data related to a sample together. Since the sample contains both the
ground truth (the first three layers) and all the backup images for the generator
(the following layers), they are split in two different tensors during training.
"""

import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import sys


def my_dataset(root, class_to_idx):
    """
    This function creates the dataset as a set of directories paths that point
    to each sample folder. In this way the samples are opened when needed and
    only their path is always saved in memory.

    Arg:
        root (str): Root directory path.
        class_to_idx (dict): dictionary with samples names as keys and ideces.

    Returns:
        list: list of (sample path, index of the sample) tuples.
    """

    imgs_folders = []
    #dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        item = (os.path.join(root, target), class_to_idx[target])
        imgs_folders.append(item)
    return imgs_folders


class GAN_dataset(VisionDataset):
    """
    Data loader for the GAN thesis needs. The sample is a merge of more images,
    both optic RGB and gray scale SAR. The images are all .jpg and processed
    with PIL.

    The folders are arranged in this way:
        root/train0/images(6 in this case)
        root/train1/images
        ...
        root/validation0/images
        root/validation1/images
        ...

    Args:
        root (str): is the dataroot path.
        extension, transform and target_transform are not used in this case.

    Attributes:
        imgs (list): stores the names fo the images; the order is made to allow
            an easier slicing later, putting the gorund truth as first image.
        classes (list): list of class names, int this case list of the samples names.
        class_to_idx (dict): stores the couple items (class, class index).
        samples (list): list of (sample path, class index) tuples.
        targets (list): The class_index value for each image in the dataset.
    """

    def __init__(self, root, extensions=None, transform=None,
                 target_transform=None):
        super(GAN_dataset, self).__init__(root, transform=transform,
                                          target_transform=target_transform)

        self.imgs = ['o1.jpg', 'o0.jpg', 's0VH.jpg', 's0VV.jpg', 's1VH.jpg', 's1VV.jpg']
        classes, class_to_idx = self._find_classes(self.root)
        samples = my_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset. In this case the "classes" are
        the names of the folders since each samples is a group of images.

        Args:
            dir (string): Root directory path, in my case it's dataroot with
                train or validation and the region it is going to learn.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir),
                and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        When an item is needed, this function is called. The image is loaded
        with PIL and, if it is a SAR image, it is converted to alpha value to
        have one layer only. Then the image is transformed to a tensor and
        normalized in the range [-1:1]. Images are concatenated together at
        the end.

        Args:
            index (int): index value.

        Returns:
            tuple: (sample, target) where sample is a 10-channels tensor with
                all the pictures related to a tensor loaded and merged together,
                and target is the index of the sample.
        """

        path, target = self.samples[index]
        for n, img in enumerate(self.imgs):
            image = Image.open(os.path.join(path,img))
            if n>1:
                image = transforms.Grayscale().__call__(image)
            image = transforms.ToTensor().__call__(image)
            mean = []
            std = []
            for l in range(image.shape[0]):
                mean.append(0.5)
                std.append(0.5)
            image = transforms.Normalize(mean=mean, std=std).__call__(image)
            if n==0:
                sample = image
                continue
            sample = torch.cat((sample, image))
        return sample, target

    def __len__(self):
        return len(self.samples)

