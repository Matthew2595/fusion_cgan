# MTcGAN for Data Fusion
Master thesis project for a multi temporal cGAN for data fusion on Sentinel's images.

The study moves from [Multi-temporal sentinel-1 and -2 data fusion for optical Image Simulation](http://arxiv.org/abs/1807.09954).

This repository contains the all code needed, with the processing of data too:
- py folder contains the main code, for pre-processing of images, networks building and training;
- plot_data folder has the required scripts used to create the training graphs and plot images for the paper copy;
- docker folder contains the Dockerfile (not tested yet) for a potential deployment on Jetson Xavier.

## Abstract
The growing interest in dual-use satellite systems and the recent advancements in machine learning open new potential remote sensing applications. Data fusion is also a task of interest since it can produce high-value products. This work moves from similar research and investigates the implementation of deep learning algorithms to accomplish a data fusion task on Sentinel imagery. The objective is to simulate an optical image using multi-temporal SAR-optical images. The tests aimed to prove the functionality of the chosen method across different datasets and how performance can change with the quantity and quality of training. The method used successfully created a simulated image that resembles its original counterpart, but the result is still blurred and relatively effective. The tests prove that hardware limitations can affect the method’s potential since larger images and deep networks need computational power. Moreover, the training behavior of GANs is still an open problem and needs to be stabilized to achieve better performance. Finally, the dataset quality is essential and can affect the performance both negatively and positively.
