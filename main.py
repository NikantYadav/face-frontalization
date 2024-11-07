from __future__ import print_function
import time
import math
import random
import os
from os import listdir
from os.path import join
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

from data import ImagePipeline  # Make sure ImagePipeline is adapted for CPU
import network

# Set random seed for reproducibility
np.random.seed(42)
random.seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)

# Where is your training dataset at?
datapath = 'training_set'

# Use CPU for training (no GPU)
device = torch.device("cpu")

# Initialize the data pipeline (ImagePipeline must work on CPU now)
train_pipe = ImagePipeline(datapath, image_size=128, random_shuffle=True, batch_size=30)
m_train = train_pipe.epoch_size()
print("Size of the training set: ", m_train)

# We will need a custom iterator for CPU-based training
train_pipe_loader = train_pipe  # directly use the CPU-based train_pipe

# Generator:
netG = network.G().to(device)
netG.apply(network.weights_init)

# Discriminator:
netD = network.D().to(device)
netD.apply(network.weights_init)

# Loss function (Binary Cross Entropy loss)
criterion = nn.BCELoss()

# Optimizers for the generator and the discriminator (Adam optimizer)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)

# Create output directory if not exists
try:
    os.mkdir('output')
except OSError:
    pass

start_time = time.time()

# Let's train for 30 epochs (meaning, we go through the entire training set 30 times):
for epoch in range(30):
    
    # Keep track of the loss values for each epoch:
    loss_L1 = 0
    loss_L2 = 0
    loss_gan = 0
    
    # Your train_pipe_loader will load the images one batch at a time
    # The inner loop iterates over those batches:
    
    for i, data in enumerate(train_pipe_loader, 0):
        
        # These are your images from the current batch:
        profile = data[0]['profiles']
        frontal = data[0]['frontals']
        
        # TRAINING THE DISCRIMINATOR
        netD.zero_grad()
        real = Variable(frontal).type('torch.FloatTensor').to(device)
        target = Variable(torch.ones(real.size()[0])).to(device)
        output = netD(real)
        # D should accept the GT images
        errD_real = criterion(output, target)
        
        profile = Variable(profile).type('torch.FloatTensor').to(device)
        generated = netG(profile)
        target = Variable(torch.zeros(real.size()[0])).to(device)
        output = netD(generated.detach()) # detach() because we are not training G here
        
        # D should reject the synthetic images
        errD_fake = criterion(output, target)
        
        errD = errD_real + errD_fake
        errD.backward()
        # Update D
        optimizerD.step()
        
        # TRAINING THE GENERATOR
        netG.zero_grad()
        target = Variable(torch.ones(real.size()[0])).to(device)
        output = netD(generated)
        
        # G wants to :
        # (a) have the synthetic images be accepted by D (= look like frontal images of people)
        errG_GAN = criterion(output, target)
        
        # (b) have the synthetic images resemble the ground truth frontal image
        errG_L1 = torch.mean(torch.abs(real - generated))
        errG_L2 = torch.mean(torch.pow((real - generated), 2))
        
        errG = GAN_factor * errG_GAN + L1_factor * errG_L1 + L2_factor * errG_L2
        
        loss_L1 += errG_L1.item()
        loss_L2 += errG_L2.item()
        loss_gan += errG_GAN.item()
        
        errG.backward()
        # Update G
        optimizerG.step()
    
    if epoch == 0:
        print('First training epoch completed in ', (time.time() - start_time), ' seconds')
    
    # Reset the DALI iterator for the next epoch
    train_pipe_loader.reset()

    # Print the absolute values of the losses to the screen:
    print('[%d/30] Training absolute losses: L1 %.7f ; L2 %.7f BCE %.7f' % ((epoch + 1), loss_L1 / m_train, loss_L2 / m_train, loss_gan / m_train))

    # Save the inputs, outputs, and ground truth frontals to files:
    vutils.save_image(profile.data, 'output/%03d_input.jpg' % epoch, normalize=True)
    vutils.save_image(real.data, 'output/%03d_real.jpg' % epoch, normalize=True)
    vutils.save_image(generated.data, 'output/%03d_generated.jpg' % epoch, normalize=True)

    # Save the pre-trained Generator as well
    torch.save(netG, 'output/netG_%d.pt' % epoch)
