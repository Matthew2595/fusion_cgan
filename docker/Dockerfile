# MIT License
# Copyright (c) 2020 Matteo Ingrosso

FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

# Create the directory for the project and proper tree folder
RUN mkdir /workspace
RUN mkdir /workspace/dataset
RUN mkdir /workspace/results

# Set directory as default
WORKDIR /workspace

# Get scripts from github and move to appropriate folder
RUN git clone https://github.com/Matthew2595/fusion_cgan.git
RUN mv /workspace/fusion_cgan/py /workspace
RUN mv /workspace/fusion_cgan/README.md /workspace
RUN rm -r /workspace/fusion_cgan



