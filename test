#!/usr/bin/env python

import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from utils import *
from data import get_training_set, get_test_set
from models import Generator, Discriminator, FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=str, help='train & test set images folder')
parser.add_argument('--outputs', type=str, default='output', help='output dir')
parser.add_argument('--threads', type=int, default=12, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--upFactor', type=int, default=4, help='low to high resolution scaling factor')
parser.add_argument('--crop_size', type=int, default=80, help='the low resolution image size')
parser.add_argument('--train_jpeg', type=int, default=0, help='jpeg image quality range(1-15). Default=10')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='checkpoints/generator.pth', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='checkpoints/discriminator.pth', help="path to discriminator weights (to continue training)")

opt = parser.parse_args()
print('===> Parameters')
print(opt)

try:
    os.makedirs(opt.outputs)
except OSError:
    pass


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Equivalent to un-normalizing ImageNet (for correct visualization)
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

print('===> Loading datasets')
test_set = get_test_set(opt.inputs, opt.crop_size, opt.upFactor, opt.train_jpeg)
dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False, drop_last=True)

generator = torch.load(opt.generatorWeights)
discriminator = torch.load(opt.discriminatorWeights)

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

target_real = Variable(torch.ones(opt.batchSize,1))
target_fake = Variable(torch.zeros(opt.batchSize,1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    target_real = target_real.cuda()
    target_fake = target_fake.cuda()

print '===> Test starting'
mean_generator_content_loss = 0.0
mean_generator_adversarial_loss = 0.0
mean_generator_total_loss = 0.0
mean_discriminator_loss = 0.0

# Set evaluation mode (not training)
generator.eval()
discriminator.eval()

avg_psnr = 0
for i, batch in enumerate(dataloader):
    if i == 10: # we only see (i+1)*batch samples
        break
    # Generate data
    low_res, high_res_real = batch
    
    # Generate real and fake inputs
    high_res_real = Variable(high_res_real)
    low_res = Variable(low_res)
    if opt.cuda:
        high_res_real = high_res_real.cuda()
        low_res = low_res.cuda()
    
    high_res_fake = generator(low_res)
    
    psnr = calcPSNR(unnormalize(high_res_fake.data), unnormalize(high_res_real.data))
    avg_psnr += psnr
    
    ######### Test discriminator #########

    discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                            adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
    mean_discriminator_loss += discriminator_loss.data[0]

    ######### Test generator #########

    real_features = Variable(feature_extractor(high_res_real).data)
    fake_features = feature_extractor(high_res_fake)

    generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
    mean_generator_content_loss += generator_content_loss.data[0]
    generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), target_real)
    mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

    generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
    mean_generator_total_loss += generator_total_loss.data[0]

    ######### Status and display #########
    sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (i+1, len(dataloader),
    discriminator_loss.data[0], generator_content_loss.data[0], generator_adversarial_loss.data[0], generator_total_loss.data[0]))

    for j in range(opt.batchSize):
        save_image(unnormalize(high_res_real.data[j]), '{}/{:04d}x{}_GT.png'.format(opt.outputs, i*opt.batchSize+j, opt.upFactor)) 
        save_image(unnormalize(high_res_fake.data[j]), '{}/{:04d}x{}_SR.png'.format(opt.outputs, i*opt.batchSize+j, opt.upFactor))
        save_image(unnormalize(low_res.data[j]), '{}/{:04d}x{}_LR.png'.format(opt.outputs, i*opt.batchSize+j, opt.upFactor))

sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (i+1, len(dataloader),
mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))
print('Avg. PSNR: {:.4f} dB'.format(avg_psnr / 10))
