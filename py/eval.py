# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

This is the code for the evaluation of the performance of the network using
images measurements from Pytorch-metrics (Hiroaki Go and Chokurei).
"""

from config import *

import os
import time
import torch
import torchvision.transforms as transforms
from metrics import SSIM, PSNR

def evaluate(dataloader, D, G, sample_size, device, now, region):
    """
    This function is used to evaluate the performance of the model.
    It is not a validation funtion, but a test one because uses a test dataset
    different from the training one and wants to get information on how well
    the model perform. Hyperparameters are taken from the paper from which the
    idea comes.
    The process is similar to training. Data is taken from dataloader, split
    and then noise is added to create the multi-layer input sample for the
    generator and the ground truth for the discriminator. The prediction is
    normalized in the range [0:1] because G does not have sigmoid layer.
    The functions for the performance data run and data is saved in the
    correct list. Images are saved in lists too.

    Args:
        dataloader (obj): this is the dataloader that loads the test dataset
        D (obj): discriminator model, acts as a function
        G (obj): generator model, acts as a function
        sample_size (int): width and height of the samples
        device (obj): type of device used to process data, used to exploit gpus
        now (str): stores the name for the folder where to save images
    """

    # Lists from config file are loaded to save performance data
    # Modules saved as variables to run evaluation indeces
    global eval_data
    psnr = PSNR()
    ssim = SSIM()

    i = 0
    start = time.time()
    # To not use the gradient attribute
    with torch.no_grad():
        # Cycle over the dataloader
        print('^^^^^^^^^^^^^^^^')
        print('>>>TEST PHASE<<<')
        print('Evaluation loop on region' + region)
        for img, _ in dataloader:
            # Get back to three dimensions
            img = img.squeeze(0).to(device)
            if i % 10 == 0 or i==1:
                print('Validation on sample ', i)

            # Ground truth directly prepared for comparison (range [0:1])
            y = img[:3,:,:].to(device)
            y = transforms.Normalize(mean=[-1,-1,-1],std=[2,2,2]).__call__(y)
            y = y.unsqueeze(0).to(device)
            #print('Memory allocated:', torch.cuda.memory_allocated())

            # Backup images
            x = img[3:,:,:].to(device)
            x = x.unsqueeze(0).to(device)
            #print('Memory allocated:', torch.cuda.memory_allocated())

            # Prediction normalized in range [0:1]
            y_pred = G(x).cuda()
            y_pred = y_pred.squeeze(0).to(device)
            y_pred = transforms.Normalize(mean=[-1,-1,-1],std=[2,2,2]).__call__(y_pred)
            y_pred = y_pred.unsqueeze(0).to(device)
            #print('Memory allocated:', torch.cuda.memory_allocated())

            # Data set in local variables
            psnr_value = psnr(y_pred, y).item()
            ssim_value = ssim(y_pred, y).item()

            # Data showed during test
            if i % 10 == 0:
                print('Iter: [%d/%d]\tPSNR: %.4f\tSSIM: %.4f\t'
                      % (i, len(dataloader), psnr_value, ssim_value))

            # Data saved in lists
            eval_data['psnr_list'].append(psnr_value)
            eval_data['ssim_list'].append(ssim_value)

            # Images saved in the correct folder
            y = y.squeeze(0)
            y_pred = y_pred.squeeze(0)

            f = transforms.ToPILImage().__call__(y_pred.detach().cpu())
            f.save(os.path.join(save_path, region, now, switch[1], 'fake', 'save'+str(i)+'.jpg'))
            r = transforms.ToPILImage().__call__(y.detach().cpu())
            r.save(os.path.join(save_path, region, now, switch[1], 'real', 'save'+str(i)+'.jpg'))

            i += 1
    end = time.time()
    print('Elapsed time in minutes:', (end - start) / 60)
    