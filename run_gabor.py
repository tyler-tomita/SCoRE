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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def run_gabor():
    # define training and validation data sets
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
    subset_data = True
    if subset_data:
        dataset_size = {
            'train':int(len(mydata['train'])*2/4),
            'val':int(len(mydata['val'])*2/4),
            }
        for phase in ['train', 'val']:
            unique_task_ids = mydata[phase].task_ids.unique()
            sample_size = int(dataset_size[phase]/len(unique_task_ids))
            subset_inds = torch.tensor([], dtype=torch.int64)
            for task_id in unique_task_ids:
                task_inds = torch.where(mydata[phase].task_ids == task_id)[0]
                perm = torch.randperm(len(task_inds))
                subset_inds = torch.cat((subset_inds, task_inds[perm[:sample_size]]))
            subset = Subset(mydata[phase], subset_inds)
            subset.task_ids = mydata[phase].task_ids[subset_inds]
            mydata[phase] = subset

    else:
        dataset_size = {
            'train':int(len(mydata['train'])),
            'val':int(len(mydata['val']))
        }
    samples_per_task = int(dataset_size['train']/4)

    # input_size = 8192 # size of input after feature extraction
    input_size = 16 * 16 * 32
    # input_size = 32 * 32 * 16
    # input_size = 128

    # define hyperparameters
    score_hyper = {
        'learning_rate_experts':5e-4, # learning rate of new expert
        'learning_rate_unfreeze':5e-4, # learning rate of old experts after unfreezing
        'learning_rate_ensembler':1e-2,
        'learning_rate_td':0.6,
        'learning_rate_context_td':0.2,
        'context_reward_weight':0.0,
        'weight_decay':0.0,
        'epochs_ensembler_check':30, # check if the ensembler retuning performance is acceptable after this many training examples
        'perf_slope_check':0.01, # if the average increase in accuracy per trial is at least this amount, then unfreeze experts
        'perf_check_ensembler':0.5, # if ensembler retuning achieves this amount above initial task accuracy (during context recognizer phase), then don't create new expert
        'perf_check_context_recognizer':0.90, # if context_recognizer achieves this accuracy, then don't create a new ensembler
        'acc_window':20,
        'mix_ratio':0.,
        'n_try_old':10, # number of initial trials to assess old task sets
        'new_task_threshold':0.35,
        'check_new_task_acc':0.95, # accuracy must be at least this high to start checking for a new task
        'replay_after_new':False,
        'amsgrad':False,
        'lambda_autoencoder':0.,
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
        'expert_replay_prob':1.
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
                # nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                # nn.ReLU(),
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
                # nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                # nn.ReLU(),
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
                nn.Linear(self.input_size, self.hidden_size, bias=True),
                nn.ReLU(),
                # nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                # nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size, bias=True),
                nn.LogSoftmax(dim=-1)
            ).to(device)

        def forward(self, inputs):
            return self.net(inputs)

    # class Expert(nn.Module):
    #     def __init__(self):
    #         super(Expert, self).__init__()
    #         self.net = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3), # hw/2
    #                                  nn.ReLU(),
    #                                  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # hw/4
    #                                  nn.ReLU(),
    #                                  nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # hw/8
    #                                  nn.ReLU(),
    #                                  nn.AdaptiveAvgPool2d((1, 1)),
    #                                  nn.Flatten() # (64)
    #                                  )
    #         self.output_size = 64

    #     def forward(self, inputs):
    #         return self.net(inputs)
        

    # class ExpertDecoder(nn.Module):
    #     def __init__(self):
    #         super(ExpertDecoder, self).__init__()

        
    #         self.net = nn.Sequential(nn.Upsample((int(img_size/8), int(img_size/8))),
    #                                  nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
    #                                  nn.ReLU(),
    #                                  nn.Upsample(scale_factor=2),
    #                                  nn.ConvTranspose2d(8, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
    #                                  nn.Sigmoid()
    #                                  )
            
    #     def forward(self, inputs):
    #         n, num_channels = inputs.shape
    #         return self.net(inputs.view(n, num_channels, 1, 1))
        

    # class Ensembler(nn.Module):
    #     def __init__(self, input_size):
    #         super(Ensembler, self).__init__()

    #         self.input_size = input_size
    #         self.output_size = output_size

    #         self.net = nn.Sequential(
    #             nn.Linear(self.input_size, self.output_size),
    #             nn.LogSoftmax(dim=-1)
    #         ).to(device)

    #     def forward(self, inputs):
    #         return self.net(inputs)
        

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
        
    hidden_size_expert = 128
    output_size_expert = 128
    # hidden_size_expert = 32
    # output_size_expert = 32
    hidden_size_ensembler = 128
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
        

    # feature_extractor = GaborFeatureExtractor().to(device)
    # feature_extractor.load_state_dict(torch.load('models/gabor-feature-extractor.pt'))
    # feature_extractor = feature_extractor.base
    feature_extractor = GaborFeatureExtractorAE().to(device)
    feature_extractor.load_state_dict(torch.load('models/gabor-dataset-v2-autoencoder.pt'))
    feature_extractor = nn.Sequential(feature_extractor.conv1, feature_extractor.conv2, feature_extractor.conv3, nn.Flatten())
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


    task_order = [0, 1, 2, 3]
    num_tasks = len(task_order)

    # run_algs = ['SCORE', 'Fine-Tune', 'EWC', 'DynaMoE']
    run_algs = ['SCORE']
    for trial in range(num_trials):
        print(f'\n##### Trial {trial+1}/{num_trials} #####')

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
        # score_model = SCORE2(Expert,
        #                      Ensembler).to(device)
        expert_decoder = ExpertDecoder(output_size_expert,
                                       hidden_size_expert,
                                       input_size).to(device)
        # expert_decoder = ExpertDecoder().to(device)
        mem_model = NEM(input_size, Encoder, Decoder, max_memories).to(device)
        # mem_model = NEM2((3, img_size, img_size), Expert, max_memories).to(device)
        dyna_model = DynaMoE(DynaExpert,
                             input_size,
                             hidden_size_expert,
                             output_size_expert,
                             output_size_ensembler
                             ).to(device)
        finetune_model = EWC(input_size,
                            hidden_size_expert,
                            output_size_expert,
                            hidden_size_ensembler,
                            2,
                            3,
                            num_tasks).to(device) # EWC but we don't regularize
        ewc_model = EWC(input_size,
                       hidden_size_expert,
                       output_size_expert,
                       hidden_size_ensembler,
                       2,
                       3,
                       num_tasks).to(device)       
        

        if 'SCORE' in run_algs:
            print()
            print('Train SCORE')
            print('-' * 15)
            train_loss['SCORE'][trial, :], train_acc['SCORE'][trial, :], _ = trainSCORE(score_model,
                                                                                        mem_model,
                                                                                        feature_extractor,
                                                                                        dataloaders['train'],
                                                                                        torch.optim.Adam,
                                                                                        score_hyper,
                                                                                        mem_hyper,
                                                                                        expert_decoder=expert_decoder,
                                                                                        task_start_inds=[i*samples_per_task for i in range(num_tasks)])
        
        if 'DynaMoE' in run_algs:
            print()
            print('Train DynaMoE')
            print('-' * 15)
            train_loss['DynaMoE'][trial, :], train_acc['DynaMoE'][trial, :] = trainDynaMoE(dyna_model,
                                                                                           feature_extractor,
                                                                                           dataloaders['train'],
                                                                                           torch.optim.Adam,
                                                                                           score_hyper,
                                                                                           task_start_inds=[i*samples_per_task for i in range(num_tasks)]
                                                                                           )
        
        if 'Fine-Tune' in run_algs:
            print()
            print('Train Fine-Tune')
            print('-' * 15)
            train_loss['Fine-Tune'][trial, :], train_acc['Fine-Tune'][trial, :] = trainSimple(finetune_model,
                                                                                              feature_extractor,
                                                                                              dataloaders['train'],
                                                                                              fineTune_hyper
                                                                                              )
        if 'EWC' in run_algs:
            print()
            print('Train EWC')
            print('-' * 15)
            for task_id in task_order:
                task_str = 'Task ' + str(task_id+1)
                subset_inds = torch.where(mydata['train'].task_ids == task_id)[0]
                task_id_tensor = mydata['train'].task_ids[subset_inds]
                data_subset = Subset(mydata['train'], subset_inds)
                data_subset.task_ids = task_id_tensor
                dataloader_subset = DataLoader(data_subset, batch_size=batch_size, shuffle=False)
                train_loss['EWC'][trial, subset_inds], train_acc['EWC'][trial, subset_inds] = trainEWC(ewc_model,
                                                                                                       feature_extractor,
                                                                                                       dataloader_subset,
                                                                                                       ewc_hyper,
                                                                                                       exclude_tasks=[]
                                                                                                       )
                ewc_model.store_task_params(task_id)
                input_embeddings = feature_extractor(torch.cat([data_subset.dataset[i][0].unsqueeze(0) for i in subset_inds], dim=0)).unsqueeze(dim=1)
                ewc_model.fisher_information(task_id, input_embeddings)

        if 'Scratch' in run_algs:
            print()
            print('Train Scratch')
            print('-' * 15)
            for task_id in task_order:
                task_str = 'Task ' + str(task_id+1)
                subset_inds = torch.where(mydata['train'].task_ids == task_id)[0]
                task_id_tensor = mydata['train'].task_ids[subset_inds]
                data_subset = Subset(mydata['train'], subset_inds)
                data_subset.task_ids = task_id_tensor
                dataloader_subset = DataLoader(data_subset, batch_size=batch_size, shuffle=False)
                scratch_model = EWC(input_size,
                                    hidden_size_expert,
                                    output_size_expert,
                                    hidden_size_ensembler,
                                    2,
                                    num_tasks-1
                                    ).to(device)
                
                train_loss['Scratch'][trial, subset_inds], train_acc['Scratch'][trial, subset_inds] = trainSimple(scratch_model,
                                                                                                                  feature_extractor,
                                                                                                                  dataloader_subset,
                                                                                                                  fineTune_hyper)

        # Test all algs except Scratch on Tasks 1 through (num_tasks-1)
        for task_id in range(num_tasks):
            task_str = 'Task ' + str(task_id+1)
            print()
            print('Test on ' + task_str)
            subset_inds = torch.where(mydata['train'].task_ids == task_id)[0]
            task_id_tensor = mydata['train'].task_ids[subset_inds]
            data_subset = Subset(mydata['train'], subset_inds)
            data_subset.task_ids = task_id_tensor
            dataloader_subset = DataLoader(data_subset, batch_size=batch_size, shuffle=False)
            if 'SCORE' in run_algs:
                print()
                print('SCORE:')
                val_acc['SCORE'][trial, task_id], task_label_acc['SCORE'][trial, task_id] = eval_score(score_model,
                                                                                                       mem_model,
                                                                                                       dataloader_subset,
                                                                                                       feature_extractor,
                                                                                                       infer_task=False
                                                                                                       )
            if 'Fine-Tune' in run_algs:
                print()
                print('Fine-Tune:')
                val_acc['Fine-Tune'][trial, task_id] = eval_ewc(finetune_model,
                                                                dataloader_subset,
                                                                task_id,
                                                                feature_extractor
                                                                )
            if 'EWC' in run_algs:
                print()
                print('EWC:')
                val_acc['EWC'][trial, task_id] = eval_ewc(ewc_model,
                                                          dataloader_subset,
                                                          task_id,
                                                          feature_extractor
                                                          )
                
            if 'DynaMoE' in run_algs:
                print()
                print('DynaMoE:')
                val_acc['DynaMoE'][trial, task_id] = eval_dynamoe(dyna_model,
                                                                  dataloader_subset,
                                                                  feature_extractor
                                                                  )

            
        results = {
            'score_hyper':score_hyper,
            'mem_hyper':mem_hyper,
            'fineTune_hyper':fineTune_hyper,
            'ewc_hyper':ewc_hyper,
            'train_loss':train_loss,
            'train_acc':train_acc,
            'val_acc':val_acc,
            'task_label_acc':task_label_acc
        }

        save_path = f'results/task-aware-gabor-continual-learning-dataset-v2-task-order-{task_order}_score_24.pt'
        torch.save(results, save_path)

if __name__ == '__main__':
   run_gabor()


# test task inference
# task_id = 3
# subset_inds = torch.where(mydata['val'].task_ids == task_id)[0]
# task_id_tensor = mydata['val'].task_ids[subset_inds]
# data_subset = Subset(mydata['val'], subset_inds)
# data_subset.task_ids = task_id_tensor
# dataloader_subset = DataLoader(data_subset, batch_size=batch_size, shuffle=False)
# score_model.eval()
# targets_subset = []
# preds = []
# for inputs, _, targets, _ in dataloader_subset:
#     targets_subset.append(targets)
#     with torch.no_grad():
#         inputs = feature_extractor(inputs)
#         preds.append(score_model(inputs))

# all_targets = torch.cat(targets_subset, dim=0)
# all_preds = torch.cat(preds, dim=0)

# print((all_preds[:, 6:].topk(1, dim=1)[1].squeeze() == all_targets).float().mean().item())

# def getEnsemblerPreds(score_model, ensembler_idx, preds):
#     start_idx = score_model.ensembler_start_indices[ensembler_idx]
#     end_idx = score_model.ensembler_start_indices[ensembler_idx+1]
#     return preds[:, start_idx:end_idx]

# score_model.eval()
# with torch.no_grad():
#     x = feature_extractor(mydatata['train'][1][0].unsqueeze(dim=0))
#     preds, _ = score_model(x)
#     pred = getEnsemblerPreds(score_model, 0, preds)
# preds