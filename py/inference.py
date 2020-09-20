# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

This script wants to load trained models and the required images to simulate
the whole scource image, not only its patches.
This was used for a visual comparison for the thesis.
The whoel images were too large, so they were simulated by patches of size 128x128
(the same used for training), and then recomposed together.
"""

from config import *
from neural_nets import *
#from essential_functions import *
from metrics import SSIM, PSNR
indices = {'PSNR':PSNR(), 'SSIM':SSIM()}

import os
from PIL import Image
import torch
import torchvision.transforms as transforms

Image.MAX_IMAGE_PIXELS = 1000000000


'''FUNCTIONS'''

def load_imgs(path, device='cpu'):
    """
    This function creates the input. It applies the same process used by the
    custom class for training and test datasets. Images are loaded, transformed
    and stacked together.

    Args:
        path (str): where to fetch the images
        device (str): store the device where to return the stacked images

    Return:
        tensor: input tensor with stacked images and to the given device
    """

    imgs = ['o1.jpg', 'o0.jpg', 's0VH.jpg', 's0VV.jpg', 's1VH.jpg', 's1VV.jpg']

    for img in imgs:
        image = Image.open(os.path.join(path, img))
        if img[0]=='s':
            image = transforms.Grayscale().__call__(image)
        image = transforms.ToTensor().__call__(image)
        mean_std = []
        for l in range(image.shape[0]):
            mean_std.append(0.5)
        image = transforms.Normalize(mean=mean_std, std=mean_std).__call__(image).to(device)
        if imgs.index(img)==0:
            sample = image
            continue
        sample = torch.cat((sample, image))
    print('Images loaded')

    return sample.to(device)


def load_nets(n_gen, gen_features, p, ngpu, path, device):
    """
    Function to load the trained networks. It creates a new netowrk as object,
    then applies the parameters previously saved. It is possible to load the
    Discriminator too.

    Args:
        n_gen (int): input features of the Generator
        gen_features (list): list of ints, it is the number of features of each
            block of the Generator
        p (float): in range [0,1], probability of dropout to use in the nets
        ngpu (int): number of GPUs available
        path (str): where to fetch the state dict
        device (str): device where to load the model

    Return:
        object: netowrk as object with the trained parameters

    """
    netG = Gen(n_gen, gen_features, ngpu, p).to(device)
    netG.load_state_dict(torch.load(os.path.join(path, 'netG_model.py')))

#    netD = Dis(n_dis, dis_features, ngpu).to(device)
#    netD.load_state_dict(torch.load(os.path.join(path, 'netD_model.py')))

    print('Model loaded')
    return netG#, netD


def forward(sample, G, imgs_size, indices, device):
    """
    This function takes the whole input, prepares it for the forward pass,
    creates the patches of given size and with two loops pass these adiacent
    patches through the generator. Then, the simulated patches are concatenaed
    together to compose the simulated source image. It stores indices values
    for performance check too.

    Args:
        sample (tensor): input previously created with stacked source images
        G (obj): Generator network with trained parameters
        imgs_size (int): it's the size of the batch (width and height) in witch
            the big picture is cropped; this number is used here to crop adiacent
            patches
        indices (dict): where the two indices are stored as objects
        device (str): device where to perform the computation

    Return:
        tensors: the two images, simulated and original, returned as tensors
            of three channels (RGB) on the given device
    """

    global eval_data
    psnr = indices['PSNR']
    ssim = indices['SSIM']

    # Split the input
    y = sample[:3,:,:]
    y = y.to(device)
    x = sample[3:,:,:]
    x = x.to(device)

    # Calculate the max number of loops to avoid going outised the image
    h_count = int(x.shape[1] / imgs_size)
    if x.shape[1]%imgs_size!=0:
        h_count += 1
    w_count = int(x.shape[2] / imgs_size)
    if x.shape[2]%imgs_size!=0:
        w_count += 1

    # First loop in height dimension
    for h in range(h_count):
        lower_h = imgs_size * h
        upper_h = imgs_size * (h+1)
        if upper_h>x.shape[1]:
            upper_h = x.shape[1]

        # Second loop for width dimension
        for w in range(w_count):
            # Croppin the image
            lower_w = imgs_size * w
            upper_w = imgs_size * (w+1)
            if upper_w>x.shape[2]:
                upper_w = x.shape[2]
            input_x = x[:, lower_h:upper_h, lower_w:upper_w].to(device)

            # Forward through the Gen
            input_x = input_x.unsqueeze(0).to(device)
            output_y = G(input_x).cuda()

            ground_truth = y[:, lower_h:upper_h, lower_w:upper_w]

            # Takes the indices
            ground_truth = ground_truth.unsqueeze(0)
            psnr_value = psnr(output_y.detach().cpu(), ground_truth.detach().cpu()).item()
            ssim_value = ssim(output_y.detach().cpu(), ground_truth.detach().cpu()).item()
            eval_data['psnr_list'].append(psnr_value)
            eval_data['ssim_list'].append(ssim_value)

            output_y = output_y.squeeze(0)

            # Recompose the row
            if w==0:
                row = output_y.detach().cpu()
                continue
            row = torch.cat((row, output_y.detach().cpu()), dim=2)
            print('Row temp shape:', row.shape)

        # Recompose the img
        if h==0:
            fake_img = row
            continue
        fake_img = torch.cat((fake_img, row), dim=1)
        print('Image temp shape:', fake_img.shape)

    print('Image generated')

    return fake_img.to(device), y.to(device)


def save_eval_data(path):
    """
    Function that saves test data lists in the correct folder.
    """
    global eval_data
    for eval_list in eval_data:
        with open(os.path.join(path, eval_list+'.txt'), 'w') as filehandle:
            filehandle.writelines("%s\n" % place for place in eval_data[eval_list])
    print('>>>Test data saved correctly.<<<')


def data_eval_and_saved(fake_sample, ground_truth, indices, device='cpu'):
    """
    Function that saves the images and evaluate the PSNR and SSIM on them.
    Indices are saved in .txt files and images as .jpg.
    """
    psnr = indices['PSNR']
    ssim = indices['SSIM']

    fake_sample = transforms.Normalize(mean=[-1,-1,-1], std=[2,2,2]).__call__(fake_sample.detach().to(device))
    ground_truth = transforms.Normalize(mean=[-1,-1,-1], std=[2,2,2]).__call__(ground_truth)
    
    fake_sample = fake_sample.unsqueeze(0).to(device)
    ground_truth = ground_truth.unsqueeze(0).to(device)
    
    psnr_value = psnr(fake_sample, ground_truth).item()
    ssim_value = ssim(fake_sample, ground_truth).item()
    
    print('>>>INDICES<<<')
    print('PSNR: %.4f SSIM: %.4f MSE: %.4f' % (psnr_value, ssim_value, mse_value))
    
    file = open(os.path.join(save_path, 'inference', region, 'indeces.txt'), 'w')
    file.write('>>>INDICES<<<\n')
    file.write('PSNR: %.4f SSIM: %.4f MSE: %.4f\n' % (psnr_value, ssim_value, mse_value))
    file.close()

    save_eval_data(os.path.join(save_path, 'inference', region))

    fake_sample = fake_sample.squeeze(0).to(device)
    ground_truth = ground_truth.squeeze(0).to(device)
    
    f = transforms.ToPILImage(mode='RGB').__call__(fake_sample.detach().cpu())
    f.save(os.path.join(save_path, 'inference', region, 'fake.jpg'))
    r = transforms.ToPILImage(mode='RGB').__call__(ground_truth.detach().cpu())
    r.save(os.path.join(save_path, 'inference', region, 'real.jpg'))
    
    print('Images are saved')
    


'''RUN'''

#device='cpu'

region = regions[0]
print(region)

sample = load_imgs(os.path.join(source_folder, region))
print('Sample shape: ', sample.shape)

folder = input('Write the folder where to fetch the models: ')

G = load_nets(n_gen, gen_features, p, ngpu, os.path.join(saved_models_path, folder), device)

fake_sample, ground_truth = forward(sample, G, imgs_size, indices, device)

data_eval_and_saved(fake_sample, ground_truth, indices)
