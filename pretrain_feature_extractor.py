from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from time import sleep

from sklearn.decomposition import PCA

from PIL import Image

import matplotlib.pyplot as plt

import time

import copy

from models.models import *
from datasets import GaborDataset
from train import *
from eval import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def main():
    img_size = 128
    mytransform = Compose([
        Resize(img_size),
        ToTensor()
    ])
    
    train_data = GaborDataset(root_dir='datasets/gabor-v2',
                              generate_data=False,
                              train=True,
                              transform=mytransform)
    
    val_data = GaborDataset(root_dir='datasets/gabor-v2',
                            generate_data=False,
                            train=False,
                            transform=mytransform)
    batch_size = 128
    dataloaders = {
    'train':DataLoader(train_data, batch_size=batch_size, shuffle=True),
    'val':DataLoader(val_data, batch_size=batch_size, shuffle=True)
    }

    dataset_sizes = {
        'train':len(train_data),
        'val':len(val_data)
    }

    # task_names = ['frequency', 'orientation-weighting', 'color']

    num_epochs = 50
    lrs = torch.tensor([5e-3])
    wd_coefs = torch.tensor([1e-5])

    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss(reduction='none')

    val_losses = torch.tensor([0. for lr in lrs for wd in wd_coefs])
    hyper_idx = 0
    for lr in lrs:
        for wd in wd_coefs:
            print('=' * 10)
            print('lr = {lr:.0e}'.format(lr=lr.item()))
            print('wd = {wd:.0e}'.format(wd=wd.item()))
            print('=' * 10)
            # cnn_model = GaborFeatureExtractor(64)
            cnn_model = GaborFeatureExtractor()
            cnn_model = cnn_model.to(device)
            optimizer = optim.Adam(cnn_model.parameters(), lr=lr, weight_decay=wd)
            cnn_model, val_loss = pretrain_feature_extractor(cnn_model, dataloaders, dataset_sizes, criterion, criterion2, optimizer, num_epochs=num_epochs)
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

    model_path = f'models/gabor-feature-extractor.pt'
    torch.save(best_model.state_dict(), model_path)

    results_path = f'results/gabor-feature-extractor-results.pt'
    torch.save(results, results_path)

def pretrain_feature_extractor(model, dataloaders, dataset_sizes, criterion, criterion2, optimizer, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    # Get initial validation loss
    # Iterate over data.
    phase = 'val'
    model.eval()
    running_loss = 0.0

    for inputs, features, _, _ in dataloaders[phase]:
        inputs = inputs.to(device)
        frequency_targets = features[:, 0].to(device).float()
        orientation_targets = features[:, 1].to(device).float()
        color_targets = features[:, 2:].to(device).float() / 255.0

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.no_grad():
            frequency_outputs, orientation_outputs, color_outputs = model(inputs)
            frequency_loss = criterion(frequency_outputs.squeeze(), frequency_targets)
            orientation_loss = criterion(orientation_outputs.squeeze(), orientation_targets)
            color_loss = criterion2(color_outputs.squeeze(), color_targets).view(inputs.size(0), -1).mean(dim=1).sum()
            color_loss /= inputs.size(0)
            loss = frequency_loss + orientation_loss + color_loss

        # statistics
        running_loss += loss.item() * inputs.size(0)

    best_loss = running_loss / dataset_sizes[phase]

    print(f'Initial validation loss: {best_loss}')

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, features, _, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                frequency_targets = features[:, 0].to(device).float()
                orientation_targets = features[:, 1].to(device).float()
                color_targets = features[:, 2:].to(device).float() / 255.0
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    frequency_outputs, orientation_outputs, color_outputs = model(inputs)
                    frequency_loss = criterion(frequency_outputs.squeeze(), frequency_targets)
                    orientation_loss = criterion(orientation_outputs.squeeze(), orientation_targets)
                    color_loss = criterion2(color_outputs.squeeze(), color_targets).view(inputs.size(0), -1).mean(dim=1).sum()
                    color_loss /= inputs.size(0)
                    loss = frequency_loss + orientation_loss + color_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'val':
                print(f'Epoch {epoch+1}/{num_epochs} validation loss: {epoch_loss}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print()
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Val Loss: {best_loss:4f}')
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss

if __name__ == '__main__':
    main()