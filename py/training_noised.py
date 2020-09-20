# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Script for the training of the model with added instance noise.
"""

from config import *
from instance_noise import *

import os
import time
import torch
import torchvision.transforms as transforms

torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''TRAINING'''

def training_with_noise(dataloader, D, optimD, G, optimG, criterion, epochs, sample_size, device, now, folder):
    """
    This is the function used for the training of the cGAN. It takes all the
    arguments needed for the training.
    At first global variables are taken into account to save traininf data.
    It loops through epochs and samples in the dataloader. Sample data is
    managed to create the correct variables with the ground truth for the
    discriminator and the multi-layer input for the generator. The training is
    accomplish using mini-batches. The discriminator is trained first, then the
    generator.
    The losses for the training are evalutated indipendently, but are added
    togethere before calling the optimization of the networks. The discriminator
    is trained on BCE with logit loss, and the losses for real and fake labels
    are calculated separately and added together. The generator is trained on
    BCE loss (only for max(log(D(G(z))))) and L1 loss; they are added together.
    At the end of each iteration training data are saved in the correct list
    and sample images are written on disk.
    
    Args:
        dataloader (obj): this is the dataloader that loads the training dataset
        D (obj): discriminator model, acts as a function
        optimD (obj): optimizer for discriminator
        G (obj): generator model, acts as a function
        optimG (obj): optimizer for discriminator
        criterion (func, or dict): loss function, in this case dictionary with
            two losses to make the loss for the project
        epochs (int): numbers of epochs
        samples_size (int): width and height of the samples
        device (obj): type of device used to process data, used to exploit gpus
        now (str): stores the name for the folder where to save images
        folder
    """

    global losses
    global real_label
    global fake_label

    print("Starting Training Loop on region:" + folder)
    for epoch in range(epochs):
        st = time.time()
        print('^^^^^^^^^^^^^^')
        print('EPOCH: ', epoch+1)
        i = 0
        for img, _ in dataloader:
            img = img.squeeze(0).to(device)
            if i % 10 == 0:
                print('Training on sample ', i)

            #print('Memory allocated:', torch.cuda.memory_allocated())

            # Ground truth
            y = img[:3,:,:]
            y = y.to(device)

            # Input images
            x = img[3:,:,:]
            x = x.to(device)
            #print('Memory allocated:', torch.cuda.memory_allocated())

            # Adding one dimension to make the training work
            y = y.unsqueeze(0).to(device)
            x = x.unsqueeze(0).to(device)
            #print('Memory allocated:', torch.cuda.memory_allocated())

            ############################
            # Training with mini-batches
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            optimD.zero_grad()  # Clean the gradient

            # Training D net for real batches
            D_real = D(y).cuda()  # D prediction of real image
            # Evaluates the loss for real images
            D_real_loss = criterion['BCE'](D_real, real_label).cuda()
            # Calulates the gradient
            D_real_loss.backward(retain_graph=True)
            # Define the mean loss for real images
            Dy = D_real_loss.mean().item()
            #print('Memory allocated:', torch.cuda.memory_allocated())

            # Training D net for fake batches
            # Generating the sample
            Gx = G(x).cuda()
            # D prediction of fake image
            D_fake = D(Gx).cuda()
            # Evaluates the loss for fake images
            D_fake_loss = criterion['BCE'](D_fake, fake_label).cuda()
            # Calulates the gradient
            D_fake_loss.backward(retain_graph=True)
            # Define the mean loss for fake images
            DGx_fake = D_fake_loss.mean().item()
            # Evaluates the total loss by adding them together
            # i.e. max(log(D(x))) and min(log(D(G(z))))
            D_loss = D_fake_loss + D_real_loss
            # Update the params of the net
            optimD.step()
            #print('Memory allocated:', torch.cuda.memory_allocated())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            optimG.zero_grad()
            # New D prediction of fake image
            D_fake = D(Gx).cuda()
            # Evaluates the BCEloss for G, i.e. max(log(D(G(z))))
            G_BCE_loss = criterion['BCE'](D_fake, real_label).cuda()
            G_BCE_loss.backward(retain_graph=True)
            # Define the mean loss of G to better fool D
            DGx_real = G_BCE_loss.mean().item()
            # Evaluates the L1 loss between the fake image and the real one
            G_L1_loss = criterion['L1'](Gx, y).cuda()
            G_L1_loss.backward(retain_graph=True)
            G_L1 = G_L1_loss.mean().item()
            G_loss = G_BCE_loss + lamda * G_L1_loss
            optimG.step()
            #print('Memory allocated:', torch.cuda.memory_allocated())

            # Output training stats
            if i % 10 == 0:
                print('Epoch: [%d/%d] Iter: [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t D(x): %.4f\tD(G(z)) (fake/real): %.4f / %.4f\tG L1: %.4f'
                      % (epoch + 1, epochs, i, len(dataloader),
                         D_loss.item(), G_loss.item(), Dy, DGx_fake, DGx_real, G_L1))

            # Save Losses for plotting later
            losses['G_BCE_losses'].append(G_BCE_loss.item())
            losses['G_L1_losses'].append(G_L1_loss.item())
            losses['G_losses'].append(G_loss.item())
            losses['D_fake_losses'].append(D_fake_loss.item())
            losses['D_real_losses'].append(D_real_loss.item())
            losses['D_losses'].append(D_loss.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 10 == 0) or ((epoch == epochs- 1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    sample = x.to(device)
                    sample = G(sample)  # to add the right input
                    sample = sample.squeeze(0).to(device)
                    sample = sample.detach().cpu()
                    sample = transforms.Normalize(mean=[-1,-1,-1], std=[2,2,2]).__call__(sample)
                y = y.squeeze(0).to(device)
                y = y.detach().cpu()
                y = transforms.Normalize(mean=[-1,-1,-1], std=[2,2,2]).__call__(y)

                f = transforms.ToPILImage().__call__(sample.cpu())
                f.save(os.path.join(save_path, folder, now, switch[0], 'fake', 'save'+str((i+train_samples*epoch))+'.jpg'))
                r = transforms.ToPILImage().__call__(y.cpu())
                r.save(os.path.join(save_path, folder, now, switch[0], 'real', 'save'+str((i+train_samples*epoch))+'.jpg'))
                del sample
                del f
                del r

            i += 1
        en = time.time()
        print('Epoch', epoch+1, 'lasted for:', str((en-st)/60), 'minutes.')
    print('Hurray!!! Training completed!!!')