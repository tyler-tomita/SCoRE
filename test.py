from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import time
from time import sleep
import math
from scipy import ndimage as ndi
import copy
from models.models import *
from datasets import GaborDataset
from train import *
from eval import *
from utils import *
from importlib import reload
from matplotlib import pyplot as plt

# class mymodel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_modules):
#         super(mymodel, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.fc1 = nn.Linear(input_size, hidden_size * num_modules)
#         self.fc2 = nn.Linear(hidden_size * num_modules, output_size * num_modules)
#         self.relu = nn.ReLU()
#         self.out = nn.Linear(output_size * num_modules, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x[:, self.hidden_size:] = 0.0
#         x = self.relu(x)
#         x = self.fc2(x)
#         x[:, self.output_size:] = 0.0
#         x = self.relu(x)
#         x = self.out(x)
#         return x
    
# model = mymodel(3, 5, 2, 2)

# x = torch.randn((10, 3))
# target = 10 * x

# out = model(x)

# criterion = nn.MSELoss()

# loss = criterion(out, target)
# loss.backward()



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

subset_inds = torch.arange(1024)
subset_inds = torch.arange(2048/2)
subset_inds = torch.arange(int(2048/2))
for phase in ['train', 'val']:
    subset = Subset(mydata[phase], subset_inds)
    subset.task_ids = mydata[phase].task_ids[subset_inds]
    mydata[phase] = subset

dataset_size = {
    'train':int(len(mydata['train'])),
    'val':int(len(mydata['val'])),
    }


samples_per_task = int(dataset_size['train']/2)

input_size = 32 * 32 * 16

# define hyperparameters
score_hyper = {
    'learning_rate_experts':7e-5, # learning rate of new expert
    'learning_rate_unfreeze':0e-4, # learning rate of old experts after unfreezing
    'learning_rate_ensembler':1e-2,
    'learning_rate_td':0.6,
    'learning_rate_context_td':0.2,
    'context_reward_weight':0.0,
    'weight_decay':0.0,
    'epochs_ensembler_check':20, # check if the ensembler retuning performance is acceptable after this many training examples
    'perf_slope_check':0.01, # if the average increase in accuracy per trial is at least this amount, then unfreeze experts
    'perf_check_ensembler':0.5, # if ensembler retuning achieves this amount above initial task accuracy (during context recognizer phase), then don't create new expert
    'perf_check_context_recognizer':0.90, # if context_recognizer achieves this accuracy, then don't create a new ensembler
    'acc_window':20,
    'mix_ratio':0.,
    'n_try_old':5, # number of initial trials to assess old task sets
    'new_task_threshold':0.35,
    'check_new_task_acc':0.95, # accuracy must be at least this high to start checking for a new task
    'replay_after_new':False,
    'amsgrad':False,
    'lambda_autoencoder':1.0,
    'reuse_ensembler_threshold':0.7,
    'max_experts':3
}

mem_hyper = {
    'k':1,
    'learning_rate':1e-2,
    'margin':1.,
    'epochs':250,
    'replay_prob_encoder':1.,
    'replay_prob_decoder':1.,
    'expert_replay_prob':0.
}

fineTune_hyper = {
    'learning_rate_experts':5e-4,
    'learning_rate_ensembler':5e-4,
    'amsgrad':False
}

ewc_hyper = {
    'learning_rate_experts':5e-4,
    'learning_rate_ensembler':5e-4,
    'lambda_ewc':1e6,
    'amsgrad':False
}    

# define and initialize the SCoRE model
class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size, bias=True),
            nn.ReLU()
        ).to(device)

    def forward(self, inputs):
        return self.net(inputs)
    
class ExpertDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ExpertDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size, bias=True),
        ).to(device)

    def forward(self, inputs):
        return self.net(inputs)
    
class Ensembler(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Ensembler, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.net = nn.Sequential(
            # nn.Linear(self.input_size, self.hidden_size, bias=True),
            # nn.ReLU(),
            # nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            # nn.ReLU(),
            nn.Linear(self.input_size, self.output_size, bias=True),
            nn.LogSoftmax(dim=-1)
        ).to(device)

    def forward(self, inputs):
        return self.net(inputs)

class DynaExpert(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size):
        super(DynaExpert, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size_2, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.output_size),
            nn.LogSoftmax(dim=-1)
        ).to(device)

    def forward(self, inputs):
        return self.net(inputs)
    
hidden_size_expert = 32
output_size_expert = 16
hidden_size_ensembler = 0
output_size_ensembler = 2

output_size = input_size
# latent_size = 100 # size of compressed memory representations
latent_size = 256
max_memories = 128 # maximum number of memories allowed per task

# define and initialize neural episodic memory model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size, bias=True),
            nn.Tanh(),
            nn.Linear(self.latent_size, self.latent_size, bias=True)
        )

    def forward(self, inputs):
        return self.net(inputs)
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size, bias=True),
            nn.Tanh(),
            nn.Linear(self.latent_size, self.output_size, bias=True)
        )

    def forward(self, inputs):
        return self.net(inputs)
    

feature_extractor = GaborFeatureExtractor().to(device)
feature_extractor.load_state_dict(torch.load('models/gabor-feature-extractor.pt'))
feature_extractor = nn.Sequential(feature_extractor.conv1, feature_extractor.conv2, nn.Flatten())
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.eval()
# feature_extractor=None

# run experiment
num_trials = 10
num_tasks = 4
train_loss = {}
train_loss = {}
train_acc = {}
val_acc = {}
task_label_acc = {'SCORE':torch.zeros((num_trials, num_tasks))}
for alg in ['SCORE', 'DynaMoE', 'Fine-Tune', 'EWC', 'Scratch']:
    train_loss[alg] = torch.zeros((num_trials, dataset_size['train']))
    train_acc[alg] = torch.zeros((num_trials, dataset_size['train']))
    if alg not in ['Scratch']:
        val_acc[alg] = torch.zeros((num_trials, num_tasks))


task_order = [0, 1]
num_tasks = len(task_order)

trial = 0

# randomly shuffle data per task
task_data = {}
for phase in ['train', 'val']:
    sample_size = int(dataset_size[phase]/num_tasks)
    shuffle_inds = torch.tensor([], dtype=torch.int64)
    for task_id in task_order:
        task_inds = torch.where(mydata[phase].task_ids == task_id)[0]
        perm = torch.randperm(len(task_inds))
        shuffle_inds = torch.cat((shuffle_inds, task_inds[perm[:sample_size]]))
    task_data[phase] = Subset(mydata[phase], shuffle_inds)

# set up dataloaders which could change if the dataset is shuffled each trial
batch_size = 128
dataloaders = {
    'train':DataLoader(task_data['train'], batch_size=batch_size, shuffle=False),
    'val':DataLoader(task_data['val'], batch_size=batch_size, shuffle=False)
    }        

score_model = SCORE(Expert,
                    Ensembler,
                    input_size,
                    hidden_size_expert,
                    output_size_expert,
                    hidden_size_ensembler,
                    output_size_ensembler).to(device)
expert_decoder = ExpertDecoder(output_size_expert,
                                hidden_size_expert,
                                input_size).to(device)
mem_model = NEM(input_size, Encoder, Decoder, max_memories).to(device)

train_loss['SCORE'][trial, :], train_acc['SCORE'][trial, :], _ = trainSCORE(score_model,
                                                                            mem_model,
                                                                            feature_extractor,
                                                                            dataloaders['train'],
                                                                            torch.optim.Adam,
                                                                            score_hyper,
                                                                            mem_hyper,
                                                                            expert_decoder=expert_decoder,
                                                                            task_start_inds=[i*samples_per_task for i in range(2)])


preds1 = []
preds2 = []
all_labels = []
score_model.eval()
all_task_inds = []
for inputs, _, labels, task_inds in dataloaders['val']:
    with torch.no_grad():
        inputs = feature_extractor(inputs)
        all_labels.append(labels)
        all_task_inds.append(task_inds)
        preds1.append(score_model.experts[0](inputs))
        preds2.append(score_model.experts[1](inputs))

preds = torch.cat((torch.cat(preds1, dim=0), torch.cat(preds2, dim=0)), dim=1)
labels = torch.cat(all_labels)
task_inds = torch.cat(all_task_inds)

inds1 = (labels == 0) & (task_inds == 0)
inds2 = (labels == 1) & (task_inds == 0)
plt.plot(preds[inds1, 0], preds[inds1, 1], '.b')
plt.plot(preds[inds2, 0], preds[inds2, 1], '.r')
plt.show()
