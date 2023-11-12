from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from time import sleep

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
    
    mydata = {}
    mydata['train'] = GaborDataset(root_dir='datasets/gabor-v2',
                                generate_data=False,
                                train=True,
                                transform=mytransform)

    mydata['val'] = GaborDataset(root_dir='datasets/gabor-v2',
                                generate_data=False,
                                train=False,
                                transform=mytransform)
    
    # randomly subset data to size `dataset_size` if desired
    subset_data = False
    if subset_data:
        dataset_sizes = {
            'train':int(len(mydata['train'])/2),
            'val':int(len(mydata['val'])/2),
            }
        for phase in ['train', 'val']:
            unique_task_ids = mydata[phase].task_ids.unique()
            sample_size = int(dataset_sizes[phase]/len(unique_task_ids))
            subset_inds = torch.tensor([], dtype=torch.int64)
            for task_id in unique_task_ids:
                task_inds = torch.where(mydata[phase].task_ids == task_id)[0]
                perm = torch.randperm(len(task_inds))
                subset_inds = torch.cat((subset_inds, task_inds[perm[:sample_size]]))
            subset = Subset(mydata[phase], subset_inds)
            subset.task_ids = mydata[phase].task_ids[subset_inds]
            mydata[phase] = subset

    else:
        dataset_sizes = {
            'train':int(len(mydata['train'])),
            'val':int(len(mydata['val']))
        }

    batch_size = 128
    dataloaders = {
        'train':DataLoader(mydata['train'], batch_size=batch_size, shuffle=False),
        'val':DataLoader(mydata['val'], batch_size=batch_size, shuffle=False)
        }

    num_epochs = 50
    # lrs = torch.tensor([1e-5, 1e-4, 1e-3])
    # wd_coefs = torch.tensor([1e-5, 1e-4, 1e-3])
    lrs = torch.tensor([5e-4])
    wd_coefs = torch.tensor([1e-5])

    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(lrs)]
    plot_markers = ['o', 's', '*', 'D', '^']
    plot_style = []
    for i in range(len(lrs)):
        for j in range(len(wd_coefs)):
            plot_style.append({'color':plot_colors[i], 'marker':plot_markers[j]})

    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss(reduction='none')

    val_losses = torch.tensor([0. for lr in lrs for wd in wd_coefs])
    hyper_idx = 0
    for lr in lrs:
        for wd in wd_coefs:
            print('lr = {lr:.0e}'.format(lr=lr.item()))
            print('wd = {wd:.0e}'.format(wd=wd.item()))
            cnn_model = GaborFeatureExtractorAE()
            cnn_model = cnn_model.to(device)
            optimizer = optim.Adam(cnn_model.parameters(), lr=lr, weight_decay=wd)
            cnn_model, val_loss, _, _, _, _ = pretrain_autoencoder(cnn_model, dataloaders, dataset_sizes, criterion, criterion2, optimizer, num_epochs=num_epochs, plot_max=1., plot_style=plot_style[hyper_idx])
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

    model_path = f'models/gabor-dataset-v2-autoencoder.pt'
    torch.save(best_model.state_dict(), model_path)

    results_path = f'results/gabor-dataset-v2-autoencoder-results.pt'
    torch.save(results, results_path)

def pretrain_autoencoder(model, dataloaders, dataset_sizes, criterion, criterion2, optimizer, num_epochs=10, plot_max = 1, plot_style=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    # Get initial validation loss
    # Iterate over data.
    phase = 'val'
    model.eval()
    running_loss = 0.0
    running_frequency_loss = 0.0
    running_orientation_loss = 0.0
    running_color_loss = 0.0
    running_reconstruction_error = 0.0

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
            frequency_outputs, orientation_outputs, color_outputs, decoder_outputs = model(inputs)
            frequency_loss = criterion(frequency_outputs.squeeze(), frequency_targets)
            orientation_loss = criterion(orientation_outputs.squeeze(), orientation_targets)
            color_loss = criterion2(color_outputs.squeeze(), color_targets).view(inputs.size(0), -1).mean(dim=1).sum()
            color_loss /= inputs.size(0)
            reconstruction_error = criterion2(decoder_outputs.to(device), inputs).view(inputs.size(0), -1).mean(dim=1).sum()
            reconstruction_error = reconstruction_error / inputs.size(0)
            loss = frequency_loss + orientation_loss + color_loss + reconstruction_error

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_frequency_loss += frequency_loss.item() * inputs.size(0)
        running_orientation_loss += orientation_loss.item() * inputs.size(0)
        running_color_loss += color_loss.item() * inputs.size(0)
        running_reconstruction_error += reconstruction_error.item() * inputs.size(0)

    best_loss = running_loss / dataset_sizes[phase]
    best_frequency_loss = running_frequency_loss / dataset_sizes[phase]
    best_orientation_loss = running_orientation_loss / dataset_sizes[phase]
    best_color_loss = running_color_loss / dataset_sizes[phase]
    best_reconstruction_error = running_reconstruction_error / dataset_sizes[phase]

    print(f'Total validation loss: {best_loss}')
    print(f'Frequency validation loss: {best_frequency_loss}')
    print(f'Orientation validation loss: {best_orientation_loss}')
    print(f'Color validation loss: {best_color_loss}')
    print(f'Reconstruction validation error: {best_reconstruction_error}')

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_frequency_loss = 0.0
            running_orientation_loss = 0.0
            running_color_loss = 0.0
            running_reconstruction_error = 0.0

            # Iterate over data.
            for inputs, features, _, _ in dataloaders[phase]:
                frequency_targets = features[:, 0].to(device).float()
                orientation_targets = features[:, 1].to(device).float()
                color_targets = features[:, 2:].to(device).float() / 255.0

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    frequency_outputs, orientation_outputs, color_outputs, decoder_outputs = model(inputs)
                    frequency_loss = criterion(frequency_outputs.squeeze(), frequency_targets)
                    orientation_loss = criterion(orientation_outputs.squeeze(), orientation_targets)
                    color_loss = criterion2(color_outputs.squeeze(), color_targets).view(inputs.size(0), -1).mean(dim=1).sum()
                    color_loss /= inputs.size(0)
                    reconstruction_error = criterion2(decoder_outputs.to(device), inputs).view(inputs.size(0), -1).mean(dim=1).sum()
                    reconstruction_error = reconstruction_error / inputs.size(0)
                    loss = frequency_loss + orientation_loss + color_loss + reconstruction_error

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_frequency_loss += frequency_loss.item() * inputs.size(0)
                running_orientation_loss += orientation_loss.item() * inputs.size(0)
                running_color_loss += color_loss.item() * inputs.size(0)
                running_reconstruction_error += reconstruction_error.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_frequency_loss = running_frequency_loss / dataset_sizes[phase]
            epoch_orientation_loss = running_orientation_loss / dataset_sizes[phase]
            epoch_color_loss = running_color_loss / dataset_sizes[phase]
            epoch_reconstruction_error = running_reconstruction_error / dataset_sizes[phase]

            if phase == 'val':
                print(f'Epoch {epoch+1}/{num_epochs} total validation loss: {epoch_loss}')
                print(f'Epoch {epoch+1}/{num_epochs} frequency validation loss: {epoch_frequency_loss}')
                print(f'Epoch {epoch+1}/{num_epochs} orientation validation loss: {epoch_orientation_loss}')
                print(f'Epoch {epoch+1}/{num_epochs} color validation loss: {epoch_color_loss}')
                print(f'Epoch {epoch+1}/{num_epochs} reconstruction validation error: {epoch_reconstruction_error}')

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
                best_frequency_loss = epoch_frequency_loss
                best_orientation_loss = epoch_orientation_loss
                best_color_loss = epoch_orientation_loss
                best_reconstruction_error = epoch_reconstruction_error
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print()
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Val Loss: {best_loss:4f}')
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_frequency_loss, best_orientation_loss, best_color_loss, best_reconstruction_error

if __name__ == '__main__':
    main()