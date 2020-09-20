# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Script used for transfer learning, takes pre-trained model and train it
on a new dataset.
"""

from config import *
from my_dataset import *
from neural_nets import *
from training import *
from eval import *
#from essential_functions import *

from datetime import datetime
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


'''def FUNCTIONS'''

def load_dataset(region):
    """
    Datasets for the new training loaded.
    """

    global data_folder
    global switch
    global batch_size

    train_dataset = GAN_dataset(os.path.join(data_folder, switch[0], region))
    eval_dataset = GAN_dataset(os.path.join(data_folder, switch[1], region))

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader



def load_nets(n_gen, gen_features, p, n_dis, dis_features, ngpu, path):
    """
    Function that encloses the loading of the models.
    The networks are created and trained parameters applied.
    The two networks are returned as objects.
    """

    netG = Gen(n_gen, gen_features, ngpu, p).to(device)
    netG.load_state_dict(torch.load(os.path.join(path, 'netG_model.py')))

    netD = Dis(n_dis, dis_features, ngpu).to(device)
    netD.load_state_dict(torch.load(os.path.join(path, 'netD_model.py')))
    print('G and D loaded')

    return netG, netD


def log_loading(log_path, region):
    """
    This function takes the log file where the region of previous training
    are written and check if the model has already seen this region.
    """
    list = []
    with open(os.path.join(log_path, 'transfer_log.txt'), 'r') as filehandle:
        for line in filehandle:
            item = line[:-1]
            list.append(item)
    
    if region in list:
        print('This is the current model log: ', list)
        raise ValueError('Ehi, the model has already learned this region, check the requested region.')

    return list, str(len(list))


def log_saving(path, regions_list):
    """
    This function rewrite the log file for the model adding the new training
    region.
    """
    global new_region
    
    regions_list.append(new_region[2:])
    
    file = open(os.path.join(path, 'transfer_log.txt'), 'wt')
    for item in regions_list:
        file.writelines(item+'\n')
    file.close()


def info_file(path):
    """
    Function that creates a .txt info file with some data regards 
    """
    file = open(os.path.join(path, 'info.txt'), 'wt')
    file.write('Test for case 1.5\n')
    file.write('Data from region %s\n' % (new_region[2:]))
    file.write('Training for %d epochs, samples size %d\n' % (epochs, imgs_size))
    file.write('Training dataset with %d samples, evaluation dataset with %d samples\n' % (train_samples, eval_samples))
    file.close()


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


def save_train_data(path):
    """
    Function that saves training data lists in the correct folder.
    """
    global losses
    for loss_list in losses:
        with open(os.path.join(path, loss_list+'.txt'), 'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in losses[loss_list])
    print('>>>Training data saved correctly.<<<')


def save_eval_data(path):
    """
    Function that saves test data lists in the correct folder.
    """
    global eval_data
    for eval_list in eval_data:
        with open(os.path.join(path, eval_list+'.txt'), 'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in eval_data[eval_list])
    print('>>>Test data saved correctly.<<<')


'''TRANSFER RUN'''

folder = input("Write the folder in '\saved models' to fetch the nets: ")
TRANSFER_PATH = os.path.join(saved_models_path, folder)
new_region = input("What region to tranfer ('0_North', '1_Centre', '2_South'): ")

transfer_log, trans_count = log_loading(TRANSFER_PATH, new_region[2:])

print("Training for %d epochs, training dataset %d samples, evaluation dataset %d samples, device %s" 
      % (epochs, train_samples, eval_samples, device.type))

# The test folder is named according to date and time of the test
now = datetime.now().strftime("%d-%m-%YT%H.%M")
# The correct three folder to save data is created
try:
    os.mkdir(os.path.join(save_path, 'transfer'+trans_count, now))
    for selection in switch:
        os.mkdir(os.path.join(save_path, 'transfer'+trans_count, now, selection))
        os.mkdir(os.path.join(save_path, 'transfer'+trans_count, now, selection, 'real'))
        os.mkdir(os.path.join(save_path, 'transfer'+trans_count, now, selection, 'fake'))
    info_file(os.path.join(save_path, 'transfer'+trans_count, now))
except FileExistsError:
    pass

# Starting point
start = time.time()

train_dataloader, eval_dataloader = load_dataset(new_region)

netG, netD = load_nets(n_gen, gen_features, p, n_dis, dis_features, ngpu, TRANSFER_PATH)
print(netG)
print(netD)

# Initializes the Binary Cross Entropy  and L1 loss functions
criterion = {'BCE':nn.BCEWithLogitsLoss().to(device), 'L1':nn.L1Loss().to(device)}

# Setup Adam optimizers for both the nets, with learning rate and other values
optimG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

netG.train()
netD.train()

training(train_dataloader, netD, optimD, netG, optimG, criterion, epochs, imgs_size, device, now, 'transfer'+trans_count)

elapsed_time(start)

torch.save(netG.state_dict(), os.path.join(save_path, 'transfer'+trans_count, now, 'netG_model.py'))
torch.save(netD.state_dict(), os.path.join(save_path, 'transfer'+trans_count, now, 'netD_model.py'))
save_train_data(os.path.join(save_path, 'transfer'+trans_count, now))

netG.eval()
netD.eval()

evaluate(eval_dataloader, netD, netG, imgs_size, device, now, 'transfer'+trans_count)

save_eval_data(os.path.join(save_path, 'transfer'+trans_count, now))

log_saving(os.path.join(save_path, 'transfer'+trans_count, now), transfer_log)
print('TRANSFER DONE')
