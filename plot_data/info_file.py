# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Matteo Ingrosso

Test script for a log file.
"""

import os
region='questa'

img_path = 'write a folder here'

file=open(os.path.join(img_path,'info.txt'), 'wt')
file.write('test case \n')
file.write('data region %s \n' % (region))
file.write('epochs %d and sample size %d \n' % (1,2))
file.write('dataset1 %d and dataset2 %d \n' % (500,200))
file.close()

lines = ['north','centre','south']
file=open(os.path.join(img_path,'lines_info.txt'), 'wt')
#file.writelines(lines+'\n')
for l in lines:
    file.writelines(l+'\n')
file.close()


def loading(path):
    this=[]
    that=[]
    with open(os.path.join(path, 'lines_info.txt'), 'r') as filehandle:
        for line in filehandle:
            current=line[:-1]
            this.append(current)
            that.append(current)
    return this, that

list1, list2=loading(img_path)
