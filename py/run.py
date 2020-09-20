# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

This is the software for the neural network for the master thesis.
"""

'''LIBRARIES'''

from config import *
from my_dataset import *
from neural_nets import *
from training import *
#from training_noised import *
from eval import *
#from essential_functions import *

from datetime import datetime
import matplotlib.pyplot as plt
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
#import torchvision.utils as vutils
#import torchvision.transforms as transforms

torch.set_default_tensor_type('torch.cuda.FloatTensor')


'''ESSENTIAL FUNCTIONS'''


def weights_init(m):
    """
    Borrowed from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    Author: Nathan Inkawhich © Copyright 2017, PyTorch

    Function for custom weights initialization. According to the reference,
    the weights should be randomly initialized from a Normal distribution with
    mean=0 and stdev=0.02. It takes an initialized model as input and reinitializes
    all convolutional and Batch normalization layers to meet this criteria. It
    is applied to the models immediately after initialization.
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        print('Convolutional init completed!')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        print('Batch norm init completed!')


def elapsed_time(start):
    """
    The function that evaluates how much time the training takes.
    The start point is taken at the beginning of the program. This function
    takes the end time and calculate the difference between start and end.
    The difference is then printed as minutes and hours.

    Args:
        start (float): it is the start time in seconds from the library
    """

    end = time.time()
    mins = (end - start) / 60
    print('Elapsed time in minutes:', mins)
    print('Elapsed time in hours:', str(int(mins/60)) + ":" + str((mins - int(mins/60))))


def save_train_data():
    """
    Function that saves training data lists in the correct folder.
    """
    global losses
    for loss_list in losses:
        with open(os.path.join(save_path, region, now, loss_list+'.txt'), 'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in losses[loss_list])
    print('>>>Training data saved correctly.<<<')


def save_eval_data():
    """
    Function that saves test data lists in the correct folder.
    """
    global eval_data
    for eval_list in eval_data:
        with open(os.path.join(save_path, region, now, eval_list+'.txt'), 'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in eval_data[eval_list])
    print('>>>Test data saved correctly.<<<')


def info_file(time):
    """
    Function that creates a .txt info file with some data regards 
    """
    file = open(os.path.join(save_path, region, time, 'info.txt'), 'wt')
    file.write('Test for case 1\n')
    file.write('Data from region %s\n' % (region[2:]))
    file.write('Training for %d epochs, samples size %d\n' % (epochs, imgs_size))
    file.write('Training dataset with %d samples, evaluation dataset with %d samples\n' % (train_samples, eval_samples))
    file.close()


'''DATASET'''
"""The datasets for both training and validation images are created, and the 
needed transformations are applied (they become tensors to allow the nets to work.)
The dataloader is created, that is the one involved in feeding the nets.
The script allows the user to chose the region, and also the possibility to mix
the dataset for a test training on all the images; if 'mix', the datasets are
loaded in a loop just before training and evaluation.
Then, the script is commanded to work on the GPU."""

region = input("Write the requested region for training ('0_North', '1_Centre', '2_South'): ")

print("Training for %d epochs, training dataset %d samples, evaluation dataset %d samples, device %s" 
      % (epochs, train_samples, eval_samples, device.type))

train_dataset = GAN_dataset(os.path.join(data_folder, switch[0], region))
eval_dataset = GAN_dataset(os.path.join(data_folder, switch[1], region))

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size)


'''SOFTWARE'''

# The test folder is named according to date and time of the test
now = datetime.now().strftime("%d-%m-%YT%H.%M")
# The correct three folder to save data is created
try:
    os.mkdir(os.path.join(save_path, region, now))
    for selection in switch:
        os.mkdir(os.path.join(save_path, region, now, selection))
        os.mkdir(os.path.join(save_path, region, now, selection, 'real'))
        os.mkdir(os.path.join(save_path, region, now, selection, 'fake'))
    info_file(now)
except FileExistsError:
    pass

# Starting point
start = time.time()

# The two nets are created with the needed parameters and are moved to the GPU
# Then initialization is applied
netG = Gen(n_gen, gen_features, ngpu, p).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)

netD = Dis(n_dis, dis_features, ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)

# Initializes the Binary Cross Entropy  and L1 loss functions
criterion = {'BCE':nn.BCEWithLogitsLoss().to(device), 'L1':nn.L1Loss().to(device)}

# Setup Adam optimizers for both the nets, with learning rate and other values
optimG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

netG.train()
netD.train()

# It's possible to chose the training with instance noise (NOT TESTED YET)
training(train_dataloader, netD, optimD, netG, optimG, criterion, epochs, imgs_size, device, now, region)
#training_with_noise(train_dataloader, netD, optimD, netG, optimG, criterion, epochs, imgs_size, device, now, region)

elapsed_time(start)

torch.save(netG.state_dict(), os.path.join(save_path, region, now, 'netG_model.py'))
torch.save(netD.state_dict(), os.path.join(save_path, region, now, 'netD_model.py'))
save_train_data()

netG.eval()
netD.eval()

evaluate(eval_dataloader, netD, netG, imgs_size, device, now, region)

save_eval_data()

# To plot data, refer to appropriate scripts
