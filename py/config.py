# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso


This file wants to set the global variables for all the modules of the net to
to not make different variables that means the same. In this way all the
parameters canbe accessed by all the modules.
"""

# Needed for labels and to set cuda as default
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""
regions (list): is where the name of the regions are saved to generate the directory link
switch (list): sotres train and validation words to create approriate folders.
imgs_size (int): it's the size of the batch (width and height) in witch the big
    picture is cropped; it's directly connected to the random numbers for cropping.
train_samples (int): is the number of cropped images, i.e samples, created for every
    picture and used to create the training dataset
eval_samples (int): is the number of cropped images, i.e samples, created for every
    picture used to create the test dataset
data_folder (str): stores the folder path to use as root for dataset creation.
source_folder (str): stores the path for the folder where to fetch the images.
py_path (str): where .py files are stored.
save_path (str): where data from training and test are saved.
saved_models_path (str): where to fetch trained models.
imgs (dict): used to store the main images when loaded using their name and ext as keys.
cropped_imgs (dict): used to save the cropped images for each sample and save them.
batch_size (int): batch size used during training, it's defined in the DataLoader
n_gen (int): input features of the Generator
n_dis (int): input features of the Discriminator, it is 3 for colored images
ngf (int): base value to relate the depth of feature in the generator
ndf (int): base value to relate the depth of feature in the discriminator
p (float): in range [0,1], probability of dropout to use in the nets
epochs (int): number of training epochs to run
lr (float): learninf rate to use for the optimizer during training
beta1 (float): beta hyperparameter for Adam optimizer
lamda (int): it's the coefficient for L1 loss
ngpu (int): number of GPUs available
gen_features (list): list of ints, it is the number of features of each block of the Generator
dis_features (list): list of ints, it is the number of features of each block od the Discriminator
patch_size (int): size of the patch to label D's output; with patchGAN it is 
    (size of the input patch / 8 - 2)
sigma (float): it is the variation for the noise to apply if the instance noise
    methos is used; it is the initial value that decreases over time
max_iters (int): it is the max number of iterations in which we want to apply
    intance noise; after this number the noise is zero, before it depends on sigma
device (obj): device object that stores cuda if available, allowing the processing
    on gpu
real_label (tensor): torch tensor full of ones (1.0) used as  classification label
    from the Discriminator
fake_label (tensor): torch tensor full of zeros (0.0) used as  classification label
    from the Discriminator
G_BCE_losses (list): used to store BCE loss for the Generator
G_L1_losses (list): used to store L1 loss for the Generator
G_losses (list): used to store total loss for the Generator
D_fake_losses (list): used to store BCE loss on fake samples for the Discriminator
D_real_losses (list): used to store BCE loss on real samples for the Discriminator
D_losses (list): used to store total loss for the Discriminator
losses (dict): used to store all the training data to easily save them together
psnr_list (list): used to store PSNR values
ssim_list (list): used to store SSIM values
eval_data (dict): used to store all the test data to easily save them together
"""

regions = ['0_North', '1_Centre', '2_South']
switch = ['train', 'validation']

# Change these values to vary the size of the batches and the number of samples
imgs_size = 256
train_samples = 300
eval_samples = 100

# The dataset folder is saved with format: dataset_(train_samples)_(eval_samples)_(imgs_size)
data_folder = '_'.join(('D:\\Tesi\\project\\dataset', str(train_samples), str(eval_samples), ('s'+str(imgs_size))))
source_folder = 'D:\\Tesi\\Dati tesi\\SENTINEL\\Export'
py_path = 'D:\\Tesi\\project\\py'
save_path = 'D:\\Tesi\\Dati tesi\\results'
saved_models_path = 'D:\\Tesi\\Dati tesi\\saved models'

# Folders for docker
#data_folder = '/workspace/dataset'
#py_path = '/woekspace/py'
#save_path = '/workspace/results'

imgs = {'o0.jpg': 0, 's0VH.jpg': 0, 's0VV.jpg': 0, 'o1.jpg': 0,  's1VH.jpg': 0, 's1VV.jpg': 0}
cropped_imgs = {'o0.jpg': 0, 's0VH.jpg': 0, 's0VV.jpg': 0, 'o1.jpg': 0,  's1VH.jpg': 0, 's1VV.jpg': 0}

batch_size = 1
n_gen = 7
n_dis = 3
ngf = 64
ndf = 64
p = 0.5

epochs = 50
lr = 0.0002
beta1 = 0.5
lamda = 100
ngpu = 1
gen_features = [64, 128, 256, 128, 64, 3]
dis_features = [64, 128, 256, 512, 1]
patch_size = 30

sigma = 0.7
max_iters = 20000

# Creates the labels for both real and fake images
# Moves the labels for both real and fake images to the device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
real_label = torch.full((1, 1, patch_size, patch_size), 1.0)
fake_label = torch.full((1, 1, patch_size, patch_size), 0.0)
real_label = real_label.to(device)
fake_label = fake_label.to(device)

# Lists to save training data
G_BCE_losses = []
G_L1_losses = []
G_losses = []
D_fake_losses = []
D_real_losses = []
D_losses = []
losses = {'G_BCE_losses':G_BCE_losses, 'G_L1_losses':G_L1_losses, 'G_losses':G_losses,
              'D_fake_losses':D_fake_losses, 'D_real_losses':D_real_losses, 'D_losses':D_losses}

# Lists to save test data
psnr_list = []
ssim_list = []
eval_data = {'psnr_list':psnr_list, 'ssim_list':ssim_list}

