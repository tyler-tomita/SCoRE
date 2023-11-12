from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
from IPython import display
from time import sleep

from sklearn.decomposition import PCA

from PIL import Image
import cv2

from io import open
import unicodedata
import string
import re
import random

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

import time
import math

from scipy import ndimage as ndi

import copy

from models.models import *
from datasets import GaborDataset2
from train import *
from eval import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    img_size = 256
    mytransform = Compose([
        Resize(img_size),
        ToTensor()
    ])
    
    train_data = GaborDataset2(root_dir='datasets/gabor-human-experiment',
                              train=True,
                              transform=mytransform)
    
    val_data = GaborDataset2(root_dir='datasets/gabor-human-experiment',
                            train=False,
                            transform=mytransform)
    batch_size = 96
    dataloaders = {
    'train':DataLoader(train_data, batch_size=batch_size, shuffle=True),
    'val':DataLoader(val_data, batch_size=batch_size, shuffle=True)
    }

    dataset_sizes = {
        'train':len(train_data),
        'val':len(val_data)
    }

    task_names = ['frequency', 'orientation-weighting']

    for task_idx, task_name in enumerate(task_names):
        print(f'Task: {task_name}')
        num_epochs = 30
        lrs = torch.tensor([1e-4])
        wd_coefs = torch.tensor([1e-4])

        plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(lrs)]
        plot_markers = ['o', 's', '*', 'D', '^']
        plot_style = []
        for i in range(len(lrs)):
            for j in range(len(wd_coefs)):
                plot_style.append({'color':plot_colors[i], 'marker':plot_markers[j]})

        criterion = nn.MSELoss()

        val_losses = torch.tensor([0. for lr in lrs for wd in wd_coefs])
        hyper_idx = 0
        for lr in lrs:
            for wd in wd_coefs:
                # print('lr = {lr:.0e}'.format(lr=lr.item()))
                # print('wd = {wd:.0e}'.format(wd=wd.item()))
                cnn_model = GaborFeatureExtractor(img_size)
                cnn_model = cnn_model.to(device)
                optimizer = optim.Adam(cnn_model.parameters(), lr=lr, weight_decay=wd)
                cnn_model, val_loss = pretrain_feature_extractor(cnn_model, dataloaders, dataset_sizes, task_idx, criterion, optimizer, num_epochs=num_epochs, plot_max=1., plot_style=plot_style[hyper_idx])
                val_losses[hyper_idx] = val_loss
                if hyper_idx == 0:
                    best_model = cnn_model
                    best_loss = val_loss
                    best_lr = lr
                    best_wd = wd
                else:
                    if val_loss < best_loss:
                        best_model = cnn_model
                        best_loss = val_loss
                        best_lr = lr
                        best_wd = wd

                hyper_idx += 1

        results = {'lrs':lrs,
                'wd_coefs':wd_coefs,
                'val_losses':val_losses,
                }

        plt.clf()
        print(f'Best overall loss: {best_loss:4f}')
        print(f'Best learning rate: {best_lr:6f}')
        print(f'Best weight decay: {best_wd:6f}')

        model_path = f'models/gabor-{task_name}-extractor-human-experiment.pt'
        torch.save(best_model.state_dict(), model_path)

        results_path = f'results/gabor-{task_name}-extractor-human-experiment-results.pt'
        torch.save(results, results_path)

def pretrain_feature_extractor(model, dataloaders, dataset_sizes, task_idx, criterion, optimizer, num_epochs=10, plot_max = 1, plot_style=None):
    # trains a model on cifar10 with classes binarized to positive and negative labels
    # must specify a list of original labels that will be mapped to the negative and positive classes
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    # Get initial validation loss
    # Iterate over data.
    phase = 'val'
    model.eval()
    running_loss = 0.0

    for inputs, targets, _ in dataloaders[phase]:
        inputs = inputs.to(device)
        targets = targets[:, task_idx].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = torch.sigmoid(model(inputs).squeeze())
            loss = criterion(outputs, targets)

        # statistics
        running_loss += loss.item() * inputs.size(0)

    best_loss = running_loss / dataset_sizes[phase]
    plot_max = best_loss

    print(f'Initial validation loss: {best_loss}')

    # if plot_style:
    #     plot_data = [[], []] # list of x and y vals to plot
    #     plot_data[0].append(0)
    #     plot_data[1].append(best_loss)
    #     plt.axis([0, num_epochs+1, 0, plot_max])
    #     plt.xticks(list(range(num_epochs+1)))
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Mean Squared Error')
    #     plt.gcf().set_size_inches(15, 5)
    #     print(plot_data[0])
    #     print(plot_data[1])
    #     print(plot_style['color'])
    #     print(plot_style['marker'])
    #     plt.scatter(plot_data[0], plot_data[1], color=plot_style['color'], marker=plot_style['marker'])
    #     lr = optimizer.param_groups[0]['lr']
    #     wd = optimizer.param_groups[0]['weight_decay']
    #     plt.title(f'Epoch {0}/{num_epochs}\n(lr: {lr:.0e}, wd: {wd:.0e})')
    #     plt.show(block=False)
    #     plt.pause(0.001)
    #     # display.display(plt.gcf())
    #     # display.clear_output(wait=True)

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, targets, _, in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets[:, task_idx].to(device).float()
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = torch.sigmoid(model(inputs).squeeze())
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'val':
                print(f'Epoch {epoch}/{num_epochs} validation loss: {epoch_loss}')

            # if phase == 'val' and plot_style:
            #     plot_data[0].append(epoch+1)
            #     plot_data[1].append(epoch_loss)
            #     plt.scatter(plot_data[0], plot_data[1], color=plot_style['color'], marker=plot_style['marker'])
            #     plt.title(f'Epoch {epoch+1}/{num_epochs}\n(lr: {lr:.0e}, wd: {wd:.0e})')
            #     plt.show(block=False)
            #     plt.pause(0.001)
            #     # display.display(plt.gcf())
            #     # display.clear_output(wait=True)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print()
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Val Loss: {best_loss:4f}')
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss


def evaluate_cnn(model, task_idx):
    # Get validation error
    # Iterate over data.
    model.eval()
    running_error = 0.0
    for inputs, targets, _ in dataloaders['val']:
        inputs = inputs.to(device)
        targets = targets[:, task_idx].to(device)

        # forward
        with torch.no_grad():
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

        # statistics
        running_loss += loss.item() * inputs.size(0)

    val_loss = running_loss / dataset_sizes['val']

    print(f'Validation Loss: {val_loss:.4f}')
    print()

    return val_loss

if __name__ == '__main__':
    main()