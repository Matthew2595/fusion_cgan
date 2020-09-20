# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

This script plot the training and test data to create appropriate graphs.
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import os
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
SMALL_SIZE = 16
MED_SIZE = 20
BIG_SIZE = 24
rcParams['axes.titlesize'] = BIG_SIZE
rcParams['font.size'] = SMALL_SIZE

rcParams['axes.labelsize'] = BIG_SIZE
rcParams['lines.linewidth'] = 1
rcParams['lines.markersize'] = 10
rcParams['xtick.labelsize'] = SMALL_SIZE
rcParams['ytick.labelsize'] = SMALL_SIZE

'''LOADING LIST
these two loops read the data
then graphs are printed
'''

data = []
folder = input('Gimme the folder: ')

G_BCE_losses = []
G_L1_losses = []
G_losses = []
D_fake_losses = []
D_real_losses = []
D_losses = []
losses = {'G_BCE_losses':G_BCE_losses, 'G_L1_losses':G_L1_losses, 'G_losses':G_losses,
              'D_fake_losses':D_fake_losses, 'D_real_losses':D_real_losses, 'D_losses':D_losses}

psnr_list = []
ssim_list = []
mse_list = []
eval_data = {'psnr_list':psnr_list, 'ssim_list':ssim_list}


for name in losses:
    list=[]
    # open file and read the content in a list
    with open(os.path.join(folder, str(name)+'.txt'), 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]

            # add item to the list
            list.append(float(currentPlace))
        losses[name]=list
#        print(name)

for name in eval_data:
    list=[]
    # open file and read the content in a list
    with open(os.path.join(folder, str(name)+'.txt'), 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]

            # add item to the list
            list.append(float(currentPlace))
        eval_data[name]=list
#        print(name)
        
#psnr=np.array(eval_data['psnr_list'])
#ssim=np.array(eval_data['ssim_list'])

#eval_data['ssim_list'].index(ssim.min())

'''PLOTTING DATA'''

###
# Losses graphs
###
# Show and compare the losses for the Discriminator and Generator
rcParams['lines.linewidth'] = 1
plt.figure(figsize=(30, 15))
plt.title('Generator and Discriminator BCE Loss During Training')
plt.plot(losses['G_BCE_losses'], label="G BCE loss")
plt.plot(losses['D_real_losses'], label="Real label")
plt.plot(losses['D_fake_losses'], label="Fake label")
plt.xlabel('Iters')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, 'G and D BCE loss during training'))

rcParams['lines.linewidth'] = 0.5
plt.figure(figsize=(30, 15))
plt.title('Total Generator and Discriminator Loss During Training')
plt.plot(losses['G_losses'], label="Tot G")
plt.plot(losses['D_losses'], label="Tot D")
plt.xlabel('Iters')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, 'Total G and D loss during training'))

fig, ax = plt.subplots()
fig.set_figheight(15)
fig.set_figwidth(30)
ax.stackplot(range(len(losses['G_BCE_losses'])),(losses['G_L1_losses'], losses['G_BCE_losses']), labels=('G_L1_losses', 'G_BCE_losses'))
ax.legend(loc='upper left')
ax.set_title('Generator Loss During Training')
ax.set_xlabel('Iters')
ax.set_ylabel('Loss')
ax.set_ylim(0,12)
#ax.set_xlim(0,50000)
plt.tight_layout()
plt.savefig(os.path.join(folder, 'G loss during training'))

fig, ax = plt.subplots()
fig.set_figheight(15)
fig.set_figwidth(30)
ax.stackplot(range(len(losses['D_real_losses'])),(losses['D_fake_losses'], losses['D_real_losses']), labels=('D_fake_losses', 'D_real_losses'))
ax.legend(loc='upper left')
ax.set_title('Discriminator Loss During Training')
ax.set_xlabel('Iters')
ax.set_ylabel('Loss')
ax.set_ylim(0,4)
#ax.set_xlim(0,50000)
plt.tight_layout()
plt.savefig(os.path.join(folder, 'D loss during training'))

rcParams['lines.linewidth'] = 1
fig, axs = plt.subplots(3, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)
fig.set_figheight(15)
fig.set_figwidth(30)
# Plot each graph, and manually set the y tick values
axs[0].plot(losses['D_fake_losses'], label="D BCE for fake label", color='r')
axs[0].set_ylim(-0.5, 16.5)
axs[0].set_title('Generator and Discriminator BCE Loss During Training')
axs[0].legend(loc='upper left')
axs[1].plot(losses['D_real_losses'], label="D BCE for real label", color='g')
axs[1].set_ylim(-0.5, 16.5)
axs[1].set_ylabel('Loss')
axs[1].legend(loc='upper left')
axs[2].plot(losses['G_BCE_losses'], label="G BCE loss")
axs[2].set_ylim(-0.5, 16.5)
axs[2].legend(loc='upper left')
plt.xlabel('Iters')
plt.tight_layout()
plt.savefig(os.path.join(folder, 'Generator and Discriminator BCE Loss During Training stacked'))


###
# Evaluation graphs
###
rcParams['lines.linewidth'] = 2
plt.figure(figsize=(30, 15))
plt.title('PSNR')
plt.plot(eval_data['psnr_list'], label="PSNR")
psnr_mean = round(np.array(eval_data['psnr_list']).mean(), 2)
plt.axhline(psnr_mean, color='r', linestyle='dashed', linewidth=1, label='Mean value')
plt.text(plt.axes().get_xlim()[1]*-0.03, psnr_mean+plt.axes().get_ylim()[1]/150, str(psnr_mean), va='center', ha="left", color='r')
plt.xlabel('Test samples')
plt.ylabel('PSNR value')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, 'PSNR_value'))

rcParams['lines.linewidth'] = 2
plt.figure(figsize=(30, 15))
plt.title('SSIM')
plt.plot(eval_data['ssim_list'], label="SSIM")
plt.axes().set_ylim(0,1)
ssim_mean = round(np.array(eval_data['ssim_list']).mean(), 2)
plt.axhline(ssim_mean, color='r', linestyle='dashed', linewidth=1, label='Mean value')
plt.text(plt.axes().get_xlim()[1]*-0.03, ssim_mean+plt.axes().get_ylim()[1]/150, str(ssim_mean), va='center', ha="left", color='r')
plt.xlabel('Test samples')
plt.ylabel('SSIM value')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, 'SSIM_value'))

