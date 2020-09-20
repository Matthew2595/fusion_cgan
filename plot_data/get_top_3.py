# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

This script takes the top three patches per region to compare them.
"""

import os
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
SMALL_SIZE = 12
MED_SIZE = 14
BIG_SIZE = 16
rcParams['axes.titlesize'] = BIG_SIZE
rcParams['font.size'] = SMALL_SIZE

rcParams['axes.titlesize'] = MED_SIZE
rcParams['axes.labelsize'] = MED_SIZE
rcParams['lines.linewidth'] = 1
rcParams['lines.markersize'] = 10
rcParams['xtick.labelsize'] = SMALL_SIZE
rcParams['ytick.labelsize'] = SMALL_SIZE

'''LOADING LIST
This script takes the folder where all the test data are save as .txt and
print the top three values
'''

data = []
folder = input('Gimme the folder: ')


psnr_list = []
ssim_list = []
mse_list = []
eval_data = {'psnr_list':psnr_list, 'ssim_list':ssim_list}

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
        
psnr=np.array(eval_data['psnr_list'])
ssim=np.array(eval_data['ssim_list'])

psnr=np.sort(psnr)
ssim=np.sort(ssim)

psnr_ind_1 = eval_data['psnr_list'].index(psnr[-1])
psnr_ind_2 = eval_data['psnr_list'].index(psnr[-2])
psnr_ind_3 = eval_data['psnr_list'].index(psnr[-3])

ssim_1 = ssim[-1]
ssim_2 = ssim[-2]
ssim_3 = ssim[-3]
ssim_ind_1 = eval_data['ssim_list'].index(ssim_1)
ssim_ind_2 = eval_data['ssim_list'].index(ssim_2)
ssim_ind_3 = eval_data['ssim_list'].index(ssim_3)
ssim_1 = round(ssim_1,4)
ssim_2 = round(ssim_2,4)
ssim_3 = round(ssim_3,4)

print('Region %s\n PSNR best three are:\n %d\n %d\n %d\n SSIM best three are:\n %d\n %d\n %d\n'
      % (folder, psnr_ind_1, psnr_ind_2, psnr_ind_3, ssim_ind_1, ssim_ind_2, ssim_ind_3))
print(round(psnr[-1],2),round(psnr[-2],2),round(psnr[-3],2))
print(ssim_1,ssim_2,ssim_3)
