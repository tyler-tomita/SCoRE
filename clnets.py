from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

import time
import math

from scipy import ndimage as ndi



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def moving_average(x, window_size):
    x_cs = np.cumsum(x)
    n_cs = len(x_cs)
    x_ave = torch.tensor([x_cs[window_size-1]/window_size] + [(x_cs[i+window_size] - x_cs[i])/window_size for i in range(n_cs-window_size)])
    return(x_ave)

def confusionMatrix(labels, predictions, num_classes, normalize=False):
    # rows are predictions; columns are ground truth labels
    cm = torch.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = ((predictions == i) & (labels == j)).sum()

    if normalize:
        cm[:, :] = cm[:, :]/predictions.shape[0]

    return cm


class CRE(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, drop_rate):
        super(CRE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        # start with one expert
        self.num_experts = 1
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_size, self.hidden_size, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=self.drop_rate),
                    nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=self.drop_rate)
                ).to(device)
            ]
        )
        self.frozen = [False]

        # ensemble layer
        self.ensemble = nn.Linear(self.num_experts*self.hidden_size, self.num_classes, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, input_tensor):
        expert_outputs = torch.zeros((input_tensor.size(0), 1, self.num_experts*self.hidden_size)).to(device)

        for i, expert in enumerate(self.experts):
            expert_outputs[:, 0, i*self.hidden_size:(i+1)*self.hidden_size] = expert(input_tensor).squeeze()

        ensemble_output = self.softmax(self.ensemble(expert_outputs))

        return expert_outputs, ensemble_output

    def add_expert(self):
        self.experts.append(
            nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=self.drop_rate),
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=self.drop_rate)
            ).to(device)
        )
        self.num_experts += 1
        self.frozen.append(False)

        # reinitizialize ensemble network with input size increased by size of expert output

#         old_weights = self.ensemble.weight.data.clone()
#         old_bias = self.ensemble.bias.data.clone()
        self.ensemble = nn.Linear(self.num_experts*self.hidden_size, self.num_classes, bias=True).to(device)
#         self.ensemble.weight.data[:, ((self.num_experts-1)*self.hidden_size):] = 0.
#         self.ensemble.weight.data[:, :((self.num_experts-1)*self.hidden_size)] = old_weights
#         self.ensemble.bias.data = old_bias

    def freeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = False
            self.frozen[i] = True

    def freeze_ensemble(self):
        for param in self.ensemble.parameters():
            param.requires_grad = False

    def unfreeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = True
            self.frozen[i] = False

    def unfreeze_ensemble(self):
        for param in self.ensemble.parameters():
            param.requires_grad = True


class DYNAMOE(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DYNAMOE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # start with one experts
        self.num_experts = 1
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_size, self.hidden_size, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.num_classes, bias=True),
                    nn.LogSoftmax(dim=-1)
                )
            ]
        )
        self.frozen = [False]

    def forward(self, input_tensor):
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(input_tensor))

        return expert_outputs

    def add_expert(self):
        self.experts.append(
            nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_classes, bias=True),
                nn.LogSoftmax(dim=-1)
            ).to(device)
        )
        self.num_experts += 1
        self.frozen.append(False)

    def freeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = False
            self.frozen[i] = True

    def unfreeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = True
            self.frozen[i] = False


class NEM(nn.Module):
    def __init__(self, input_size, encoder_size, max_memories):
        super(NEM, self).__init__()

        self.input_size = input_size
        self.encoder_size = encoder_size

        self.n_tasks = 0
        self.max_memories = max_memories # max memories per task
        self.n_memories = {}

        self.memory_encoder = nn.Linear(self.input_size, self.encoder_size, bias=True)
        self.memory_decoder = nn.Linear(self.encoder_size, self.input_size, bias=True)

#         self.memory_encoder = nn.Sequential(
#             nn.Linear(self.input_size, self.encoder_size, bias=True),
#             nn.Tanh(),
#             nn.Linear(self.encoder_size, self.encoder_size, bias=True)
#         )
#         self.memory_decoder = nn.Sequential(
#             nn.Linear(self.encoder_size, self.encoder_size, bias=True),
#             nn.Tanh(),
#             nn.Linear(self.encoder_size, self.input_size, bias=True)
#         )

        # initialize memory buffer

        self.memories = {}
        self.memories['params'] = {}

    def forward(self, input_tensor):
        output_tensor = self.memory_encoder(input_tensor)
        return output_tensor

    def add_task(self):
        if self.n_tasks == 0:
            self.memories['input'] = torch.zeros((self.max_memories, self.input_size)).to(device)
            self.memories['compressed'] = torch.zeros((self.max_memories, self.encoder_size)).to(device)
            self.memories['target'] = torch.zeros((self.max_memories, ), dtype=torch.long).to(device)
            self.memories['task'] = torch.tensor([self.n_tasks for i in range(self.max_memories)], dtype=torch.short).to(device)
        else:
            self.memories['input'] = torch.cat((self.memories['input'], torch.zeros((self.max_memories, self.input_size), device=device)), dim=0)
            self.memories['compressed'] = torch.cat((self.memories['compressed'], torch.zeros((self.max_memories, self.encoder_size), device=device)), dim=0)
            self.memories['target'] = torch.cat((self.memories['target'], torch.zeros((self.max_memories, ), dtype=torch.long, device=device)))
            self.memories['task'] = torch.cat((self.memories['task'], torch.tensor([self.n_tasks for i in range(self.max_memories)], dtype=torch.short, device=device)))
        self.n_memories[self.n_tasks] = 0
        self.n_tasks += 1

    def add_memory(self, input_tensor, target_tensor):
        if self.n_memories[self.n_tasks-1] < self.max_memories:
            mem_idx = int((self.n_memories[self.n_tasks-1] % self.max_memories) + (self.n_tasks-1)*self.max_memories)
            self.memories['input'][mem_idx, :] = input_tensor
            self.memories['target'][mem_idx] = target_tensor
            self.n_memories[self.n_tasks-1] += 1

    def add_compressed(self, indices, new_memories):
        self.memories['compressed'][indices] = new_memories

    def store_params(self, ensemble_params):
        self.memories['params'][self.n_tasks-1] = ensemble_params

    def freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = True


class EWC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_experts, drop_rate):
        super(EWC, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_experts = num_experts

        self.experts = nn.ModuleList([])
        for i in range(self.num_experts):
            self.experts.append(
                nn.Sequential(
                    nn.Linear(input_size, hidden_size, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=drop_rate),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=drop_rate)
                )
            )
        self.ensemble = nn.Linear(self.num_experts*self.hidden_size, self.num_classes, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.F = {} # fisher info
        self.task_params = {} # state of all network parameters after each task

    def forward(self, input_tensor):
        expert_outputs = torch.zeros((input_tensor.size(0), 1, self.num_experts*self.hidden_size)).to(device)

        for i, expert in enumerate(self.experts):
            expert_outputs[:, 0, i*self.hidden_size:(i+1)*self.hidden_size] = expert(input_tensor).squeeze()

        ensemble_output = self.softmax(self.ensemble(expert_outputs))

        return ensemble_output

    def store_task_params(self, task_name):
        self.task_params[task_name] = [param.detach().clone() for param in self.parameters()]

    def fisher_information(self, task_name, train_data):
        # compute according to equation (6) in the appendix of https://arxiv.org/pdf/1904.07734.pdf
        n = train_data['input'].shape[0]

        self.F[task_name] = [torch.zeros_like(param) for param in self.task_params[task_name]]

        for i in range(n):
            self.zero_grad()

            log_likelihoods = self.forward(train_data['input'][i]).squeeze()
            argmax = log_likelihoods.topk(1)[1]
            pred = log_likelihoods[argmax]

            pred.backward()

            for layer_ind, param in enumerate(self.parameters()):
                self.F[task_name][layer_ind] += 1/n*param.grad**2


class FineTune(nn.Module):
    def __init__(self, input_size, hidden_size_expert, num_classes, num_experts, num_hidden_expert=2):
        super(FineTune, self).__init__()

        self.input_size = input_size
        self.hidden_size_expert = hidden_size_expert
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.num_hidden_expert = num_hidden_expert

        self.experts = nn.ModuleList([])
        for i in range(self.num_experts):
            expert = nn.Sequential()
            for j in range(self.num_hidden_expert):
                if j == 0:
                    in_size = self.input_size
                else:
                    in_size = self.hidden_size_expert
                expert.append(nn.Linear(in_size, self.hidden_size_expert, bias=True))
                expert.append(nn.ReLU())
            self.experts.append(expert)
            # self.experts.append(
            #     nn.Sequential(
            #         nn.Linear(input_size, hidden_size, bias=True),
            #         nn.ReLU(),
            #         nn.Linear(hidden_size, hidden_size, bias=True),
            #         nn.ReLU(),
            #     )
            # )

        # takes in the expert activations
        self.ensemble = nn.Sequential(
            nn.Linear(self.num_experts*self.hidden_size_expert, self.num_classes, bias=True),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input_tensor):
        expert_outputs = torch.zeros((input_tensor.size(0), 1, self.num_experts*self.hidden_size_expert)).to(device)
        for i, expert in enumerate(self.experts):
            expert_outputs[:, 0, i*self.hidden_size_expert:(i+1)*self.hidden_size_expert] = expert(input_tensor).squeeze()

        output_tensor = self.ensemble(expert_outputs)

        return output_tensor


class ContextGate(nn.Module):
    def __init__(self, input_size, hidden_size_expert, hidden_size_gate, num_classes, num_experts, num_hidden_expert=2, num_hidden_gate=1):
        super(ContextGate, self).__init__()

        self.input_size = input_size
        self.hidden_size_expert = hidden_size_expert
        self.hidden_size_gate = hidden_size_gate
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.num_hidden_expert = num_hidden_expert
        self.num_hidden_gate = num_hidden_gate

        self.experts = nn.ModuleList([])
        for i in range(self.num_experts):
            expert = nn.Sequential()
            for j in range(self.num_hidden_expert):
                if j == 0:
                    in_size = self.input_size
                else:
                    in_size = self.hidden_size_expert
                expert.append(nn.Linear(in_size, self.hidden_size_expert, bias=True))
                expert.append(nn.ReLU())
            self.experts.append(expert)
            # self.experts.append(
            #     nn.Sequential(
            #         nn.Linear(input_size, hidden_size, bias=True),
            #         nn.ReLU(),
            #         nn.Linear(hidden_size, hidden_size, bias=True),
            #         nn.ReLU()
            #     )
            # )
        self.heads = nn.ModuleList([])
        for i in range(self.num_experts):
          self.heads.append(
              nn.Sequential(
                  nn.Linear(self.num_experts*self.hidden_size_expert, self.num_classes, bias=True),
                  nn.LogSoftmax(dim=-1)
              )
          )
        self.num_heads = len(self.heads)

        self.gate = nn.Sequential()
        for i in range(self.num_hidden_gate):
            if i == 0:
                in_size = self.input_size
            else:
                in_size = self.hidden_size_gate
            self.gate.append(nn.Linear(in_size, self.hidden_size_gate))
            self.gate.append(nn.ReLU())
        self.gate.append(nn.Linear(self.hidden_size_gate, self.num_heads))
        self.gate.append(nn.Softmax(dim=-1))
        # self.gate = nn.Sequential(
        #     nn.Linear(self.input_size, self.gate_hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.gate_hidden_size, self.num_heads),
        #     nn.Softmax(dim=-1)
        # )

    def forward(self, input_tensor):
        gate_outputs = self.gate(input_tensor)

        expert_outputs = torch.zeros((input_tensor.size(0), 1, self.num_experts*self.hidden_size_expert)).to(device)
        for i, expert in enumerate(self.experts):
            expert_outputs[:, 0, i*self.hidden_size_expert:(i+1)*self.hidden_size_expert] = expert(input_tensor).squeeze()

        output_tensor = torch.zeros((input_tensor.size(0), 1, self.num_classes)).to(device)
        for head_idx, head in enumerate(self.heads):
            output_tensor += gate_outputs[:, :, head_idx].unsqueeze(2)*head(expert_outputs)

        return output_tensor


def trainContextGate(model, train_data, hyper, feedback_interval=1):

    learning_rate_experts = hyper['learning_rate_experts']
    learning_rate_head = hyper['learning_rate_head']
    learning_rate_gate = hyper['learning_rate_gate']

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in model.heads.parameters()], 'lr':learning_rate_head},
        {'params':[param for param in model.experts.parameters()], 'lr':learning_rate_experts},
        {'params':[param for param in model.gate.parameters()], 'lr':learning_rate_gate}
    ]
    optimizer = optim.Adam(params)
    criterion = torch.nn.NLLLoss()
    loss_list = [0.0]*feedback_interval
    optimizer.zero_grad()

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    for i in range(n_train):
        with torch.no_grad():
            pred = model(train_data['input'][i:(i+1), :, :])
            current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
        train_loss[i] = current_loss.item()
        train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

        if not (i+1) % feedback_interval:
            for j in range(feedback_interval):
                ind = i - (feedback_interval - j - 1)
                pred = model(train_data['input'][ind:(ind+1), :, :])
                loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                loss.backward()
                # print(model.gate[0].weight.grad)
                optimizer.step()
                optimizer.zero_grad()

        if not i % print_every:
            print('Context Gate {}/{} iterations complete'.format(i, n_train))

    return train_loss, train_acc

def eval_context_gate(model, input_tensor, target_tensor):
  n = input_tensor.shape[0]

  with torch.no_grad():
    preds = model(input_tensor).view(n, -1)

  acc = (preds.topk(1, dim=1)[1].squeeze() == target_tensor.squeeze()).float().mean()

  return acc


def trainBaseline(model, train_data, hyper, feedback_interval=1):

    learning_rate_experts = hyper['learning_rate_experts']
    learning_rate_head = hyper['learning_rate_head']

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in model.ensemble.parameters()], 'lr':learning_rate_head},
        {'params':[param for param in model.experts.parameters()], 'lr':learning_rate_experts}
    ]
    optimizer = optim.Adam(params)
    criterion = torch.nn.NLLLoss()
    loss_list = [0.0]*feedback_interval
    optimizer.zero_grad()

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    for i in range(n_train):
        with torch.no_grad():
            pred = model(train_data['input'][i:(i+1), :, :])
            current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
        train_loss[i] = current_loss.item()
        train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

        if not (i+1) % feedback_interval:
            for j in range(feedback_interval):
                ind = i - (feedback_interval - j - 1)
                pred = model(train_data['input'][ind:(ind+1), :, :])
                loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if not i % print_every:
            print('Baseline {}/{} iterations complete'.format(i, n_train))

    return train_loss, train_acc

def eval_baseline(model, input_tensor, target_tensor):
  n = input_tensor.shape[0]

  with torch.no_grad():
    preds = model(input_tensor).view(n, -1)

  acc = (preds.topk(1, dim=1)[1].squeeze() == target_tensor.squeeze()).float().mean()

  return acc


def trainCRE(cre_model, mem_model, train_data, cre_hyper, mem_hyper, feedback_interval=1, freeze_experts=True):

    learning_rate_experts = cre_hyper['learning_rate_experts']
    learning_rate_ensemble = cre_hyper['learning_rate_ensemble']
    learning_rate_unfreeze = cre_hyper['learning_rate_unfreeze']
    learning_rate_td = cre_hyper['learning_rate_td']
    learning_rate_context_td = cre_hyper['learning_rate_context_td']
    context_reward_weight = cre_hyper['context_reward_weight']
    perf_slope_check = cre_hyper['perf_slope_check']
    perf_check = cre_hyper['perf_check']
    epochs_ensemble_check = cre_hyper['epochs_ensemble_check']
    acc_window = cre_hyper['acc_window']
#     l2_lambda = cre_hyper['l2_lambda']
    mix_ratio = cre_hyper['mix_ratio'] # number in [0, 1] specifying how much to mix old memory ensemble with current ensemble
    n_try_old = cre_hyper['n_try_old']

    learning_rate_mem = mem_hyper['learning_rate']
    margin = mem_hyper['margin']
    decoder_epochs = mem_hyper['epochs']
    replay_prob_encoder = mem_hyper['replay_prob_encoder']
    replay_prob_decoder = mem_hyper['replay_prob_decoder']
    k = mem_hyper['k']
    expert_replay_prob = mem_hyper['expert_replay_prob']

    if mem_model.n_tasks > 2:
        use_compressed = True
        rpe = replay_prob_encoder
    else:
        use_compressed = False
        rpe = 1.

    replay_old_task = False

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in cre_model.ensemble.parameters()], 'lr':learning_rate_ensemble},
        {'params':[param for expert in cre_model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
    ]
    optimizer_cre = optim.Adam(params, lr=learning_rate_experts)
    criterion = torch.nn.NLLLoss()
    loss_list = [0.0]*feedback_interval
    optimizer_cre.zero_grad()

    keep_training = True


    # if first expert, then train the expert and the ensemble network
    if cre_model.num_experts == 1 and not cre_model.frozen[-1]:
        print('train first expert')

        train_loss = torch.zeros(n_train)
        train_acc = torch.zeros(n_train)

        for i in range(n_train):

            with torch.no_grad():
                pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = current_loss.item()
            train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
#             loss_list[i%feedback_interval] = current_loss
#             loss += current_loss/feedback_interval

            if not (i+1) % feedback_interval:
                for j in range(feedback_interval):
                    ind = i - (feedback_interval - j - 1)
                    pred = cre_model(train_data['input'][ind:(ind+1), :, :])[1]
                    loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                    loss.backward()
                    optimizer_cre.step()
                    optimizer_cre.zero_grad()

#             optimizer_cre.zero_grad()

#             pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
#             loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
#             train_loss[i] = loss.item()
#             train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

#             loss.backward()
#             optimizer_cre.step()

            # store memories
            mem_model.add_memory(train_data['input'][i], train_data['target'][i])

            if not i % print_every:
                print('CRE {}/{} iterations complete'.format(i, n_train))

        cre_model.freeze_expert([0])

        mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

    # try tuning the ensemble network
    else:
        if freeze_experts:
          cre_model.freeze_expert([i for i in range(cre_model.num_experts)])

        # initialize expected rewards for each task set and each context
        Q_task = torch.zeros(mem_model.n_tasks)
        Q_context = torch.zeros(mem_model.n_tasks)
        Q_overall = torch.zeros(mem_model.n_tasks)
        acc_list = [torch.zeros(mem_model.n_tasks)]*feedback_interval
        action_list = [0]*feedback_interval

        train_loss = torch.zeros(n_train)
        train_acc = torch.zeros(n_train)

        keep_training_ensemble = True

        i = 0

        # initialize ensemble parameters with nearest neighbor old task
        label_prob = predictNEM(mem_model, train_data['input'][i], use_compressed, k).squeeze()
        cre_model.ensemble.weight.data[:] = 0.
        cre_model.ensemble.bias.data[:] = 0.
        for task in range(mem_model.n_tasks-1):
            retrieved_weight = mem_model.memories['params'][task]['weight']
            retrieved_size = retrieved_weight.shape[1]
            retrieved_bias = mem_model.memories['params'][task]['bias']
            weight_temp = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
            weight_temp[:, :retrieved_size] = retrieved_weight
            cre_model.ensemble.weight.data += label_prob[task]*weight_temp
            cre_model.ensemble.bias.data += label_prob[task]*retrieved_bias
            Q_context[task] = (1 - learning_rate_context_td)*Q_context[task] + learning_rate_context_td*label_prob[task] # initialize tast set reward with task probability based on context
        mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

        # initialize reward of new tentative task set
        Q_context[-1] = torch.max(Q_context[:-1])

        Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

#         retrieved_weight = mem_model.memories['params'][nearest_task]['weight']
#         retrieved_size = retrieved_weight.shape[1]
#         retrieved_bias = mem_model.memories['params'][nearest_task]['bias']
#         cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
#         cre_model.ensemble.bias.data = retrieved_bias

        print('train ensemble')
        while keep_training_ensemble:

            # //////
#             if i > 0:
#                 _ = trainEncoder(mem_model, train_data['input'][i], learning_rate = learning_rate_mem, margin = margin, replay_prob = rpe, use_compressed = use_compressed)

#                 nearest_task = int(predictNEM(mem_model, train_data['input'][i], use_compressed, 1).topk(1)[1].squeeze().numpy())
#                 retrieved_weight = mem_model.memories['params'][nearest_task]['weight']
#                 retrieved_size = retrieved_weight.shape[1]
#                 retrieved_bias = mem_model.memories['params'][nearest_task]['bias']
#                 weight_temp = torch.zeros_like(cre_model.ensemble.weight.data)
#                 weight_temp[:, :retrieved_size] = retrieved_weight
#                 cre_model.ensemble.weight.data = mix_ratio*weight_temp + (1 - mix_ratio)*cre_model.ensemble.weight.data
#                 cre_model.ensemble.bias.data = mix_ratio*retrieved_bias + (1 - mix_ratio)*cre_model.ensemble.bias.data
#                 for task in range(mem_model.n_tasks):
#                     if task == nearest_task:
#                         q_update = 1.
#                     else:
#                         q_update = 0.
#                     Q_context[nearest_task] = (1 - learning_rate_context_td)*Q_context[nearest_task] + learning_rate_context_td*q_update

#             Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

            if i > 0:
                _ = trainEncoder(mem_model, train_data['input'][i], learning_rate = learning_rate_mem, margin = margin, replay_prob = rpe, use_compressed = use_compressed)

                nearest_task = int(predictNEM(mem_model, train_data['input'][i], use_compressed, 1).topk(1)[1].squeeze().numpy())

                for task in range(mem_model.n_tasks):
                    if task == nearest_task:
                        q_update = 1.
                    else:
                        q_update = 0.
                    Q_context[nearest_task] = (1 - learning_rate_context_td)*Q_context[nearest_task] + learning_rate_context_td*q_update

                Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

#                 action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
#                 best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
#                 if best_action != mem_model.n_tasks - 1:
#                     retrieved_weight = mem_model.memories['params'][int(best_action)]['weight']
#                     retrieved_size = retrieved_weight.shape[1]
#                     retrieved_bias = mem_model.memories['params'][int(best_action)]['bias']
#                     weight_temp = torch.zeros_like(cre_model.ensemble.weight.data)
#                     weight_temp[:, :retrieved_size] = retrieved_weight
#                     cre_model.ensemble.weight.data = mix_ratio*weight_temp + (1 - mix_ratio)*cre_model.ensemble.weight.data
#                     cre_model.ensemble.bias.data = mix_ratio*retrieved_bias + (1 - mix_ratio)*cre_model.ensemble.bias.data
            # //////

            if i < n_try_old:
                # select action that will give the highest expected reward
                action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
                best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
                action_list[i%feedback_interval] = best_action
#                 print('best action: ' + str(int(best_action)))
    #             state_dict = cre_model.ensemble.state_dict()
                for task in range(mem_model.n_tasks):
                    retrieved_weight = mem_model.memories['params'][task]['weight']
                    retrieved_size = retrieved_weight.shape[1]
                    cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                    cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                    cre_model.ensemble.bias.data = mem_model.memories['params'][task]['bias']
                    with torch.no_grad():
                        pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                    current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
                    current_acc = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
                    acc_list[i%feedback_interval][task] = current_acc[0]
#                     acc[task] += current_acc[0]/feedback_interval
                    if not (i+1) % feedback_interval:
                        for acc in acc_list:
                            Q_task[task] = (1 - learning_rate_td)*Q_task[task] + learning_rate_td*acc[task]
                            acc[task] = 0.0
                    if task == best_action:
                        train_loss[i] = current_loss
                        train_acc[i] = current_acc
    #                     cre_model.ensemble.load_state_dict(state_dict)
                if i == n_try_old - 1:
                    print('reinstate task set ' + str(int(best_action)))
                    retrieved_weight = mem_model.memories['params'][int(best_action)]['weight']
                    retrieved_size = retrieved_weight.shape[1]
                    cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                    cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                    cre_model.ensemble.bias.data = mem_model.memories['params'][int(best_action)]['bias']
            else:
                with torch.no_grad():
                    pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                    current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
                train_loss[i] = current_loss.item()
                train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()


            if replay_old_task:
                if torch.rand((1,)) < expert_replay_prob:
                    task_replay_prob = Q_overall[:-1]/Q_overall[:-1].sum()
                    replay_task_ind = int(torch.multinomial(task_replay_prob, 1))
                    replay_mem_ind = torch.randint(low=replay_task_ind*mem_model.max_memories, high=(replay_task_ind+1)*mem_model.max_memories-1, size=(1,))
                    replay_mem = mem_model.memories['input'][int(replay_mem_ind)]
                    pred_mem = cre_model(replay_mem)[1]
                    loss = criterion(pred_mem.view(1, -1), mem_model.memories['target'][int(replay_mem_ind)].view(-1))
                    loss.backward()
                    optimizer_cre.step()
                    optimizer_cre.zero_grad()

            if not (i+1) % feedback_interval:
                for j in range(feedback_interval):
                    ind = i - (feedback_interval - j - 1)
                    pred = cre_model(train_data['input'][ind:(ind+1), :, :])[1]
                    loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                    loss.backward()
                    optimizer_cre.step()
                    optimizer_cre.zero_grad()

#             optimizer_cre.zero_grad()

#             pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
#             loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
#             train_loss[i] = loss.item()


#             if replay_old_task and use_compressed:
#                 if torch.rand((1,)) < expert_replay_prob:
#                     task_replay_prob = Q_overall[:-1]/Q_overall[:-1].sum()
#                     replay_task_ind = int(torch.multinomial(task_replay_prob, 1))
#                     replay_mem_ind = torch.randint(low=replay_task_ind*mem_model.max_memories, high=(replay_task_ind+1)*mem_model.max_memories-1, size=(1,))
#                     replay_mem = mem_model.memory_decoder(mem_model.memories['compressed'][int(replay_mem_ind)])
#                     pred_mem = cre_model(replay_mem)[1]
#                     loss += criterion(pred_mem.view(1, -1), mem_model.memories['target'][int(replay_mem_ind)].view(-1))

#             loss.backward()
#             optimizer_cre.step()

#             if i < n_try_old:
#                 # select action that will give the highest expected reward
#                 action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
#                 best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
# #                 print('best action: ' + str(int(best_action)))
#                 if best_action == mem_model.n_tasks - 1:
#                     train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
#                     Q_task[-1] = (1 - learning_rate_td)*Q_task[-1] + learning_rate_td*train_acc[i]
#                 else:
#                     state_dict = cre_model.ensemble.state_dict()
#                     for task in range(mem_model.n_tasks-1):
#                         retrieved_weight = mem_model.memories['params'][task]['weight']
#                         retrieved_size = retrieved_weight.shape[1]
#                         cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data)
#                         cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
#                         cre_model.ensemble.bias.data = mem_model.memories['params'][task]['bias']
#                         with torch.no_grad():
#                             pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
#                             acc = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
#                             Q_task[task] = (1 - learning_rate_td)*Q_task[task] + learning_rate_td*acc
#                         if task == best_action:
#                             train_acc[i] = acc
#                     if i == n_try_old - 1:
#                         print('reinstate task set ' + str(int(best_action)))
#                         retrieved_weight = mem_model.memories['params'][int(best_action)]['weight']
#                         retrieved_size = retrieved_weight.shape[1]
#                         cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data)
#                         cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
#                         cre_model.ensemble.bias.data = mem_model.memories['params'][int(best_action)]['bias']
#                     else:
#                         cre_model.ensemble.load_state_dict(state_dict)
#             else:
#                 train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
# #             train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

            # store memories
            mem_model.add_memory(train_data['input'][i], train_data['target'][i])
            mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

            i += 1

            if not i % print_every:
                print('CRE {}/{} iterations complete'.format(i, n_train))

            if i == epochs_ensemble_check:
                perf_slope = torch.mean(train_acc[(i-acc_window):i]) - torch.mean(train_acc[:acc_window])
                perf = torch.mean(train_acc[(i-acc_window):i])
                print('acc slope: ' + str(perf_slope))
                print('acc: ' + str(perf))
                if perf < perf_check:
                # if  perf_slope < perf_slope_check and perf < perf_check:
                    keep_training_ensemble = False
                    print('ensemble performance is insufficient: recruit new expert')
                else:
                    replay_old_task = True
                    cre_model.unfreeze_expert([i for i in range(cre_model.num_experts)])
                    params = [
                        {'params':[param for param in cre_model.ensemble.parameters()],
                         'weight_decay':0.,
                         'lr':learning_rate_ensemble
                        },
                        {'params':[expert[0].bias for expert in cre_model.experts] + [expert[3].bias for expert in cre_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        },
                        {'params':[expert[0].weight for expert in cre_model.experts] + [expert[3].weight for expert in cre_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        }
                    ]
                    optimizer_cre = optim.Adam(params, lr=learning_rate_experts)
                    optimizer_cre.zero_grad()
                    print('ensemble performance is sufficient: unfreeze experts')
            elif i >= n_train:
                keep_training_ensemble = False
                keep_training = False

        if keep_training:
            print('expert added at iteration {}'.format(str(i)))
            cre_model.add_expert()

            params = [
                {'params':[param for param in cre_model.ensemble.parameters()], 'lr':learning_rate_ensemble},
                {'params':[param for expert in cre_model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
            ]
            optimizer_cre = optim.Adam(params, lr=learning_rate_experts)
            criterion = torch.nn.NLLLoss()
            optimizer_cre.zero_grad()

            print('train new expert')
            while keep_training:

                _ = trainEncoder(mem_model, train_data['input'][i], learning_rate = learning_rate_mem, margin = margin, replay_prob = rpe, use_compressed = use_compressed)

                nearest_task = int(predictNEM(mem_model, train_data['input'][i], use_compressed, k).topk(1)[1].squeeze().numpy())
                retrieved_weight = mem_model.memories['params'][nearest_task]['weight']
                retrieved_size = retrieved_weight.shape[1]
                retrieved_bias = mem_model.memories['params'][nearest_task]['bias']
                weight_temp = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                weight_temp[:, :retrieved_size] = retrieved_weight
                cre_model.ensemble.weight.data = mix_ratio*weight_temp + (1 - mix_ratio)*cre_model.ensemble.weight.data
                cre_model.ensemble.bias.data = mix_ratio*retrieved_bias + (1 - mix_ratio)*cre_model.ensemble.bias.data

                with torch.no_grad():
                    pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                    current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
                train_loss[i] = current_loss.item()
                train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
    #             loss_list[i%feedback_interval] = current_loss
    #             loss += current_loss/feedback_interval

                if not (i+1) % feedback_interval:
                    for j in range(feedback_interval):
                        ind = i - (feedback_interval - j - 1)
                        pred = cre_model(train_data['input'][ind:(ind+1), :, :])[1]
                        loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                        loss.backward()
                        optimizer_cre.step()
                        optimizer_cre.zero_grad()

#                 optimizer_cre.zero_grad()

#                 pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
#                 loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
#                 train_loss[i] = loss.item()
#                 train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

#                 loss.backward()
#                 optimizer_cre.step()

                # store memories
                mem_model.add_memory(train_data['input'][i], train_data['target'][i])
                mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

                i += 1
                if not i % print_every:
                    print('CRE {}/{} iterations complete'.format(i, n_train))

                if i >= n_train:
                    keep_training = False

        if mem_model.n_tasks == 2:
            with torch.no_grad():
                train_target = mem_model.memories['input'][:int(mem_model.n_tasks*mem_model.max_memories)]
                train_encoded = mem_model.memory_encoder(train_target)

            for epoch in range(decoder_epochs):
                _ = trainDecoder(mem_model, train_encoded, train_target)

        else:
            with torch.no_grad():
                decompressed_memories = mem_model.memory_decoder(mem_model.memories['compressed'][:int((mem_model.n_tasks-1)*mem_model.max_memories)])
                train_target = torch.cat([decompressed_memories, mem_model.memories['input'][int((mem_model.n_tasks-1)*mem_model.max_memories):int(mem_model.n_tasks*mem_model.max_memories)]], dim=0)
                train_encoded = mem_model.memory_encoder(train_target)

            for epoch in range(decoder_epochs):
                # randomly sample proportion p of old memories for training decoder
                n_sample = round(replay_prob_decoder*(mem_model.n_tasks-1)*mem_model.max_memories)
                sample_inds = random.sample([i for i in range((mem_model.n_tasks-1)*mem_model.max_memories)], n_sample)
                sample_inds += [i + (mem_model.n_tasks-1)*mem_model.max_memories for i in range(mem_model.max_memories)]
                _ = trainDecoder(mem_model, train_encoded[sample_inds], train_target[sample_inds])

        mem_model.add_compressed(range(int(mem_model.n_tasks*mem_model.max_memories)), train_encoded)

    return train_loss, train_acc

def eval_cre(cre_model, mem_model, input_tensor, target_tensor, task_label_tensor, num_tasks, use_compressed=False):
  n = input_tensor.shape[0]
  preds = torch.zeros((n, 2))
  task_label_preds = torch.zeros(n)
  unique_task_labels = torch.tensor(list(range(num_tasks)))

  for i in range(n):
    # initialize ensemble parameters with nearest neighbor old task
    task_label_preds[i] = predictNEM(mem_model, input_tensor[i], use_compressed, 1).squeeze().topk(1)[1].squeeze()
    # predicted_task_set = task_label_preds[i].topk(1)[1].squeeze()
    cre_model.ensemble.weight.data[:] = 0.
    cre_model.ensemble.bias.data[:] = 0.

    retrieved_weight = mem_model.memories['params'][int(task_label_preds[i])]['weight']
    retrieved_size = retrieved_weight.shape[1]
    retrieved_bias = mem_model.memories['params'][int(task_label_preds[i])]['bias']
    cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
    cre_model.ensemble.bias.data[:] = retrieved_bias

    with torch.no_grad():
      preds[i] = cre_model(input_tensor[i, :, :])[1]

  preds = preds.to(device)
  task_label_preds = task_label_preds.to(device)
  acc = (preds.topk(1, dim=1)[1].squeeze() == target_tensor.squeeze()).float().mean()
  task_label_acc = (task_label_preds == task_label_tensor.squeeze()).float().mean()
  task_label_confusion = confusionMatrix(task_label_tensor.squeeze(), task_label_preds, num_tasks, normalize=False)

  return acc, task_label_acc, task_label_confusion

# def predict(cre_model, input_tensor):
#     with torch.no_grad():
#         pred = cre_model(input_tensor)[1]
#     return pred

def trainTUCRE(cre_model, mem_model, train_data, cre_hyper, mem_hyper, feedback_interval=1, freeze_experts=True, use_compressed=False):

    learning_rate_experts = cre_hyper['learning_rate_experts']
    learning_rate_ensemble = cre_hyper['learning_rate_ensemble']
    learning_rate_unfreeze = cre_hyper['learning_rate_unfreeze']
    learning_rate_td = cre_hyper['learning_rate_td']
    learning_rate_context_td = cre_hyper['learning_rate_context_td']
    context_reward_weight = cre_hyper['context_reward_weight']
    perf_slope_check = cre_hyper['perf_slope_check']
    perf_check = cre_hyper['perf_check']
    epochs_ensemble_check = cre_hyper['epochs_ensemble_check']
    acc_window = cre_hyper['acc_window']
    mix_ratio = cre_hyper['mix_ratio'] # number in [0, 1] specifying how much to mix old memory ensemble with current ensemble
    n_try_old = cre_hyper['n_try_old']
    new_task_threshold = cre_hyper['new_task_threshold']
    replay_after_new = cre_hyper['replay_after_new']

    learning_rate_mem = mem_hyper['learning_rate']
    margin = mem_hyper['margin']
    decoder_epochs = mem_hyper['epochs']
    replay_prob_encoder = mem_hyper['replay_prob_encoder']
    replay_prob_decoder = mem_hyper['replay_prob_decoder']
    k = mem_hyper['k']
    expert_replay_prob = mem_hyper['expert_replay_prob']

    rpe = replay_prob_encoder

    replay_old_task = False

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in cre_model.ensemble.parameters()], 'lr':learning_rate_ensemble},
        {'params':[param for expert in cre_model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
    ]
    optimizer_cre = optim.Adam(params, lr=learning_rate_experts)
    optimizer_cre.zero_grad()
    criterion = torch.nn.NLLLoss()
    loss_list = [0.0]*feedback_interval

    # train_mode is one of 'expert', 'ensemble', or 'reinforcement'
    # start in 'expert' mode
    train_mode = 'expert'
    print('train expert')
    within_mode_idx = 0
    within_task_idx = 0
    mem_model.add_task()

    keep_training = True

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    chance_acc = (train_data['target'] == torch.mode(train_data['target'], dim=0)[0]).float().mean()
    recent_acc = (torch.rand(acc_window).to(device) < chance_acc).float()
    best_acc = chance_acc

    for i in range(n_train):
        if mem_model.n_tasks > 1:
            _ = trainEncoder(mem_model, train_data['input'][i], learning_rate = learning_rate_mem, margin = margin, replay_prob = rpe, use_compressed = use_compressed)

        if train_mode == 'expert':
            if mem_model.n_tasks > 1:
                nearest_task = int(predictNEM(mem_model, train_data['input'][i], use_compressed, k).topk(1)[1].squeeze().numpy())
                retrieved_weight = mem_model.memories['params'][nearest_task]['weight']
                retrieved_size = retrieved_weight.shape[1]
                retrieved_bias = mem_model.memories['params'][nearest_task]['bias']
                weight_temp = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                weight_temp[:, :retrieved_size] = retrieved_weight
                cre_model.ensemble.weight.data = mix_ratio*weight_temp + (1 - mix_ratio)*cre_model.ensemble.weight.data
                cre_model.ensemble.bias.data = mix_ratio*retrieved_bias + (1 - mix_ratio)*cre_model.ensemble.bias.data

            with torch.no_grad():
                pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = current_loss.item()
            train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
            recent_acc[:-1] = recent_acc[1:].clone()
            recent_acc[-1] = train_acc[i]
            # print(recent_acc)

            if not (within_task_idx+1) % feedback_interval:
                for j in range(feedback_interval):
                    ind = i - (feedback_interval - j - 1)
                    pred = cre_model(train_data['input'][ind:(ind+1), :, :])[1]
                    loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                    loss.backward()
                    optimizer_cre.step()
                    optimizer_cre.zero_grad()
                    # print(loss.item())

            within_mode_idx += 1
            within_task_idx += 1

            mean_acc = recent_acc.mean()
            if mean_acc > best_acc:
                best_acc = mean_acc
            elif best_acc - mean_acc > new_task_threshold:

                # we think a new task has been encountered

                # replay previous task memories before instantiating new task set
                if replay_after_new:
                    for mem_idx in range(mem_model.max_memories*(mem_model.n_tasks-1), mem_model.max_memories*mem_model.n_tasks):
                        replay_mem = mem_model.memories['input'][mem_idx]
                        pred_mem = cre_model(replay_mem)[1]
                        loss = criterion(pred_mem.view(1, -1), mem_model.memories['target'][mem_idx].view(-1))
                        loss.backward()
                        optimizer_cre.step()
                        optimizer_cre.zero_grad()

                cre_model.freeze_expert(list(range(cre_model.num_experts)))
                mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

                # when we think a new task is encountered, enter reinforcement mode
                train_mode = 'reinforcement'
                print('new task at iteration {}'.format(str(i)))
                print('train reinforcement')
                time.sleep(1.)
                within_mode_idx = 0
                within_task_idx = 0
                best_acc = chance_acc

                # # train decoder and add compressed memories for the most recently learned task set
                # if mem_model.n_tasks == 2:
                #     with torch.no_grad():
                #         train_target = mem_model.memories['input'][:int(mem_model.n_tasks*mem_model.max_memories)]
                #         train_encoded = mem_model.memory_encoder(train_target)

                #     for epoch in range(decoder_epochs):
                #         _ = trainDecoder(mem_model, train_encoded, train_target)

                #     mem_model.add_compressed(range(int(mem_model.n_tasks*mem_model.max_memories)), train_encoded)

                # elif mem_model.n_tasks > 2:
                #     with torch.no_grad():
                #         decompressed_memories = mem_model.memory_decoder(mem_model.memories['compressed'][:int((mem_model.n_tasks-1)*mem_model.max_memories)])
                #         train_target = torch.cat([decompressed_memories, mem_model.memories['input'][int((mem_model.n_tasks-1)*mem_model.max_memories):int(mem_model.n_tasks*mem_model.max_memories)]], dim=0)
                #         train_encoded = mem_model.memory_encoder(train_target)

                #     for epoch in range(decoder_epochs):
                #         # randomly sample proportion p of old memories for training decoder
                #         n_sample = round(replay_prob_decoder*(mem_model.n_tasks-1)*mem_model.max_memories)
                #         sample_inds = random.sample([i for i in range((mem_model.n_tasks-1)*mem_model.max_memories)], n_sample)
                #         sample_inds += [i + (mem_model.n_tasks-1)*mem_model.max_memories for i in range(mem_model.max_memories)]
                #         _ = trainDecoder(mem_model, train_encoded[sample_inds], train_target[sample_inds])

                #     mem_model.add_compressed(range(int(mem_model.n_tasks*mem_model.max_memories)), train_encoded)

                mem_model.add_task()

                # initialize expected rewards for each task set and each context
                Q_task = torch.zeros(mem_model.n_tasks)
                Q_context = torch.zeros(mem_model.n_tasks)
                Q_overall = torch.zeros(mem_model.n_tasks)
                acc_list = [torch.zeros(mem_model.n_tasks)]*feedback_interval
                action_list = [0]*feedback_interval

                # initialize ensemble parameters with nearest neighbor old task
                label_prob = predictNEM(mem_model, train_data['input'][i], use_compressed, k).squeeze()
                cre_model.ensemble.weight.data[:] = 0.
                cre_model.ensemble.bias.data[:] = 0.
                for task in range(mem_model.n_tasks-1):
                    retrieved_weight = mem_model.memories['params'][task]['weight']
                    retrieved_size = retrieved_weight.shape[1]
                    retrieved_bias = mem_model.memories['params'][task]['bias']
                    weight_temp = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                    weight_temp[:, :retrieved_size] = retrieved_weight
                    cre_model.ensemble.weight.data += label_prob[task]*weight_temp
                    cre_model.ensemble.bias.data += label_prob[task]*retrieved_bias
                    Q_context[task] = (1 - learning_rate_context_td)*Q_context[task] + learning_rate_context_td*label_prob[task] # initialize tast set reward with task probability based on context
                mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

                # initialize reward of new tentative task set
                Q_context[-1] = torch.max(Q_context[:-1])

                Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

        elif train_mode == 'reinforcement':

            nearest_task = int(predictNEM(mem_model, train_data['input'][i], use_compressed, 1).topk(1)[1].squeeze().numpy())

            for task in range(mem_model.n_tasks):
                if task == nearest_task:
                    q_update = 1.
                else:
                    q_update = 0.
                Q_context[task] = (1 - learning_rate_context_td)*Q_context[task] + learning_rate_context_td*q_update

            Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

            # select action that will give the highest expected reward
            action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
            best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
            action_list[within_task_idx%feedback_interval] = best_action
            for task in range(mem_model.n_tasks):
                retrieved_weight = mem_model.memories['params'][task]['weight']
                retrieved_size = retrieved_weight.shape[1]
                cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                cre_model.ensemble.bias.data = mem_model.memories['params'][task]['bias']
                with torch.no_grad():
                    pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
                current_acc = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
                acc_list[within_task_idx%feedback_interval][task] = current_acc[0]
                if not (within_task_idx+1) % feedback_interval:
                    for acc in acc_list:
                        Q_task[task] = (1 - learning_rate_td)*Q_task[task] + learning_rate_td*acc[task]
                        acc[task] = 0.0
                if task == best_action:
                    train_loss[i] = current_loss
                    train_acc[i] = current_acc
                    recent_acc[:-1] = recent_acc[1:].clone()
                    recent_acc[-1] = train_acc[i]
            if within_mode_idx == (n_try_old - 1):
                print('reinstate task set ' + str(int(best_action)))
                retrieved_weight = mem_model.memories['params'][int(best_action)]['weight']
                retrieved_size = retrieved_weight.shape[1]
                cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                cre_model.ensemble.bias.data = mem_model.memories['params'][int(best_action)]['bias']

                # switch to ensemble mode
                train_mode = 'ensemble'
                print('train ensemble')
                time.sleep(1.)
                within_mode_idx = 0

                # if freeze_experts:
                #   cre_model.freeze_expert([i for i in range(cre_model.num_experts)])

            else:
                within_mode_idx += 1

            mean_acc = recent_acc.mean()
            if mean_acc > best_acc:
                best_acc = mean_acc

            within_task_idx += 1

        # try tuning the ensemble network
        elif train_mode == 'ensemble':
            if mem_model.n_tasks > 1:
                nearest_task = int(predictNEM(mem_model, train_data['input'][i], use_compressed, k).topk(1)[1].squeeze().numpy())
                retrieved_weight = mem_model.memories['params'][nearest_task]['weight']
                retrieved_size = retrieved_weight.shape[1]
                retrieved_bias = mem_model.memories['params'][nearest_task]['bias']
                weight_temp = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                weight_temp[:, :retrieved_size] = retrieved_weight
                cre_model.ensemble.weight.data = mix_ratio*weight_temp + (1 - mix_ratio)*cre_model.ensemble.weight.data
                cre_model.ensemble.bias.data = mix_ratio*retrieved_bias + (1 - mix_ratio)*cre_model.ensemble.bias.data

            with torch.no_grad():
                pred = cre_model(train_data['input'][i:(i+1), :, :])[1]

            current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = current_loss.item()
            train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
            recent_acc[:-1] = recent_acc[1:].clone()
            recent_acc[-1] = train_acc[i]

            ###########################################
            # fix:
            # 1. check if Q_overall is right
            # 2. sample memory from the old task with the highest Q_overall
            ###########################################
            if replay_old_task:
                if torch.rand((1,)) < expert_replay_prob:

                    # select action that will give the highest expected reward
                    action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
                    best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
                    for task in range(mem_model.n_tasks-1):
                        retrieved_weight = mem_model.memories['params'][task]['weight']
                        retrieved_size = retrieved_weight.shape[1]
                        cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                        cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                        cre_model.ensemble.bias.data = mem_model.memories['params'][task]['bias']
                        with torch.no_grad():
                            pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                        current_acc = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
                        Q_task[task] = (1 - learning_rate_td)*Q_task[task] + learning_rate_td*current_acc
                    Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context


                    task_replay_prob = Q_overall[:-1]/Q_overall[:-1].sum()
                    # replay_task_ind = int(torch.multinomial(task_replay_prob, 1))
                    replay_task_ind = int(task_replay_prob.topk(1)[1])

                    retrieved_weight = mem_model.memories['params'][replay_task_ind]['weight']
                    retrieved_size = retrieved_weight.shape[1]
                    cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                    cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                    cre_model.ensemble.bias.data = mem_model.memories['params'][replay_task_ind]['bias']
                    replay_mem_ind = torch.randint(low=replay_task_ind*mem_model.max_memories, high=(replay_task_ind+1)*mem_model.max_memories-1, size=(1,))
                    replay_mem = mem_model.memories['input'][int(replay_mem_ind)]
                    pred_mem = cre_model(replay_mem)[1]
                    loss = criterion(pred_mem.view(1, -1), mem_model.memories['target'][int(replay_mem_ind)].view(-1))
                    loss.backward()
                    optimizer_cre.step()
                    optimizer_cre.zero_grad()

                    retrieved_weight = mem_model.memories['params'][mem_model.n_tasks-1]['weight']
                    retrieved_size = retrieved_weight.shape[1]
                    cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                    cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                    cre_model.ensemble.bias.data = mem_model.memories['params'][mem_model.n_tasks-1]['bias']

            if not (within_task_idx+1) % feedback_interval:
                for j in range(feedback_interval):
                    ind = i - (feedback_interval - j - 1)
                    pred = cre_model(train_data['input'][ind:(ind+1), :, :])[1]
                    loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                    loss.backward()
                    optimizer_cre.step()
                    optimizer_cre.zero_grad()

            within_task_idx += 1

            if within_mode_idx == (epochs_ensemble_check-1):
                perf_slope = torch.mean(train_acc[(i-acc_window):i]) - torch.mean(train_acc[:acc_window])
                perf = torch.mean(train_acc[(i-acc_window):i])
                print('acc slope: ' + str(perf_slope))
                print('acc: ' + str(perf))
                if perf < perf_check:
                # if  perf_slope < perf_slope_check and perf < perf_check:
                    print('ensemble performance is insufficient: recruit new expert at iteration {}'.format(str(i)))
                    time.sleep(1.)
                    train_mode = 'expert'
                    within_mode_idx = 0
                    cre_model.add_expert()
                    params = [
                        {'params':[param for param in cre_model.ensemble.parameters()], 'lr':learning_rate_ensemble},
                        {'params':[param for expert in cre_model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
                    ]
                    optimizer_cre = optim.Adam(params, lr=learning_rate_experts)
                    criterion = torch.nn.NLLLoss()
                    optimizer_cre.zero_grad()
                else:
                    within_mode_idx += 1
                    replay_old_task = True
                    cre_model.unfreeze_expert([i for i in range(cre_model.num_experts)])
                    params = [
                        {'params':[param for param in cre_model.ensemble.parameters()],
                         'weight_decay':0.,
                         'lr':learning_rate_ensemble
                        },
                        {'params':[expert[0].bias for expert in cre_model.experts] + [expert[3].bias for expert in cre_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        },
                        {'params':[expert[0].weight for expert in cre_model.experts] + [expert[3].weight for expert in cre_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        }
                    ]
                    optimizer_cre = optim.Adam(params)
                    optimizer_cre.zero_grad()
                    print('ensemble performance is sufficient: unfreeze experts')
                    time.sleep(1.)
            elif within_mode_idx > (epochs_ensemble_check - 1):
                mean_acc = recent_acc.mean()
                if mean_acc > best_acc:
                    best_acc = mean_acc
                elif best_acc - mean_acc > new_task_threshold:

                    # we think a new task has been encountered

                    # replay previous task memories before instantiating new task set
                    if replay_after_new:
                        for mem_idx in range(mem_model.max_memories*(mem_model.n_tasks-1), mem_model.max_memories*mem_model.n_tasks):
                            replay_mem = mem_model.memories['input'][mem_idx]
                            pred_mem = cre_model(replay_mem)[1]
                            loss = criterion(pred_mem.view(1, -1), mem_model.memories['target'][mem_idx].view(-1))
                            loss.backward()
                            optimizer_cre.step()
                            optimizer_cre.zero_grad()

                    cre_model.freeze_expert(list(range(cre_model.num_experts)))
                    mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

                    # when we think a new task is encountered, enter reinforcement mode
                    train_mode = 'reinforcement'
                    print('new task at iteration {}'.format(str(i)))
                    print('train reinforcement')
                    time.sleep(1.)
                    within_mode_idx = 0
                    within_task_idx = 0
                    best_acc = chance_acc

                    # # train decoder and add compressed memories for the most recently learned task set
                    # if mem_model.n_tasks == 2:
                    #     with torch.no_grad():
                    #         train_target = mem_model.memories['input'][:int(mem_model.n_tasks*mem_model.max_memories)]
                    #         train_encoded = mem_model.memory_encoder(train_target)

                    #     for epoch in range(decoder_epochs):
                    #         _ = trainDecoder(mem_model, train_encoded, train_target)

                    #     mem_model.add_compressed(range(int(mem_model.n_tasks*mem_model.max_memories)), train_encoded)

                    # elif mem_model.n_tasks > 2:
                    #     with torch.no_grad():
                    #         decompressed_memories = mem_model.memory_decoder(mem_model.memories['compressed'][:int((mem_model.n_tasks-1)*mem_model.max_memories)])
                    #         train_target = torch.cat([decompressed_memories, mem_model.memories['input'][int((mem_model.n_tasks-1)*mem_model.max_memories):int(mem_model.n_tasks*mem_model.max_memories)]], dim=0)
                    #         train_encoded = mem_model.memory_encoder(train_target)

                    #     for epoch in range(decoder_epochs):
                    #         # randomly sample proportion p of old memories for training decoder
                    #         n_sample = round(replay_prob_decoder*(mem_model.n_tasks-1)*mem_model.max_memories)
                    #         sample_inds = random.sample([i for i in range((mem_model.n_tasks-1)*mem_model.max_memories)], n_sample)
                    #         sample_inds += [i + (mem_model.n_tasks-1)*mem_model.max_memories for i in range(mem_model.max_memories)]
                    #         _ = trainDecoder(mem_model, train_encoded[sample_inds], train_target[sample_inds])

                    #     mem_model.add_compressed(range(int(mem_model.n_tasks*mem_model.max_memories)), train_encoded)

                    mem_model.add_task()

                    # initialize expected rewards for each task set and each context
                    Q_task = torch.zeros(mem_model.n_tasks)
                    Q_context = torch.zeros(mem_model.n_tasks)
                    Q_overall = torch.zeros(mem_model.n_tasks)
                    acc_list = [torch.zeros(mem_model.n_tasks)]*feedback_interval
                    action_list = [0]*feedback_interval

                    # initialize ensemble parameters with nearest neighbor old task
                    label_prob = predictNEM(mem_model, train_data['input'][i], use_compressed, k).squeeze()
                    cre_model.ensemble.weight.data[:] = 0.
                    cre_model.ensemble.bias.data[:] = 0.
                    for task in range(mem_model.n_tasks-1):
                        retrieved_weight = mem_model.memories['params'][task]['weight']
                        retrieved_size = retrieved_weight.shape[1]
                        retrieved_bias = mem_model.memories['params'][task]['bias']
                        weight_temp = torch.zeros_like(cre_model.ensemble.weight.data, device=device)
                        weight_temp[:, :retrieved_size] = retrieved_weight
                        cre_model.ensemble.weight.data += label_prob[task]*weight_temp
                        cre_model.ensemble.bias.data += label_prob[task]*retrieved_bias
                        Q_context[task] = (1 - learning_rate_context_td)*Q_context[task] + learning_rate_context_td*label_prob[task] # initialize tast set reward with task probability based on context
                    mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

                    # initialize reward of new tentative task set
                    Q_context[-1] = torch.max(Q_context[:-1])

                    Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context
            else:
                within_mode_idx += 1

        # store memories
        mem_model.add_memory(train_data['input'][i], train_data['target'][i])
        mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

        if not i % print_every:
            print('CRE {}/{} iterations complete'.format(i, n_train))

    return train_loss, train_acc


def trainDYNAMOE(dyna_model, train_data, dyna_hyper, feedback_interval=1):

    learning_rate_experts = dyna_hyper['learning_rate_experts']
    learning_rate_unfreeze = dyna_hyper['learning_rate_unfreeze']
    learning_rate_td = dyna_hyper['learning_rate_td'] # temporal difference learning rate
#     learning_rate_context_td = dyna_hyper['learning_rate_context_td']
#     context_reward_weight = dyna_hyper['context_reward_weight']
    perf_slope_check = dyna_hyper['perf_slope_check']
    perf_check = dyna_hyper['perf_check']
    epochs_gate_check = dyna_hyper['epochs_gate_check']
    acc_window = dyna_hyper['acc_window']
#     l2_lambda = dyna_hyper['l2_lambda']
    mix_ratio = dyna_hyper['mix_ratio'] # number in [0, 1] specifying how much to mix old memory ensemble with current ensemble
    n_try_old = dyna_hyper['n_try_old']

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for expert in dyna_model.experts for i in [0, 2, 4] for param in expert[i].parameters()], 'lr':learning_rate_experts}
    ]
    optimizer_dyna = optim.Adam(params, lr=learning_rate_experts)
    criterion = torch.nn.NLLLoss()
    loss_list = [0.0]*feedback_interval
    optimizer_dyna.zero_grad()

    keep_training = True

    # if first expert, then train the expert and the ensemble network
    if dyna_model.num_experts == 1 and not dyna_model.frozen[-1]:
        print('train first expert')

        train_loss = torch.zeros(n_train)
        train_acc = torch.zeros(n_train)

        for i in range(n_train):

            with torch.no_grad():
                pred = dyna_model(train_data['input'][i:(i+1), :, :])[0]
                current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = current_loss.item()
            train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

            if not (i+1) % feedback_interval:
                for j in range(feedback_interval):
                    ind = i - (feedback_interval - j - 1)
                    pred = dyna_model(train_data['input'][ind:(ind+1), :, :])[0]
                    loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                    loss.backward()
                    optimizer_dyna.step()
                    optimizer_dyna.zero_grad()

            if not i % print_every:
                print('DyaMoE {}/{} iterations complete'.format(i, n_train))

        dyna_model.freeze_expert([0])

    # try reusing existing expert
    else:
        dyna_model.freeze_expert([i for i in range(dyna_model.num_experts)])

        # initialize expected rewards for each expert
        Q_experts = torch.zeros(dyna_model.num_experts)
        acc_list = [torch.zeros(dyna_model.num_experts) for j in range(feedback_interval)]
        action_list = [0]*feedback_interval


        train_loss = torch.zeros(n_train)
        train_acc = torch.zeros(n_train)

        keep_training_gate = True
        train_old_expert = False

        i = 0

        # initialize ensemble parameters with nearest neighbor old task
#         label_prob = predictNEM(mem_model, train_data['input'][i], use_compressed, k).squeeze()
#         dyna_model.ensemble.weight.data[:] = 0.
#         dyna_model.ensemble.bias.data[:] = 0.
#         for task in range(mem_model.n_tasks-1):
#             retrieved_weight = mem_model.memories['params'][task]['weight']
#             retrieved_size = retrieved_weight.shape[1]
#             retrieved_bias = mem_model.memories['params'][task]['bias']
#             weight_temp = torch.zeros_like(dyna_model.ensemble.weight.data)
#             weight_temp[:, :retrieved_size] = retrieved_weight
#             dyna_model.ensemble.weight.data += label_prob[task]*weight_temp
#             dyna_model.ensemble.bias.data += label_prob[task]*retrieved_bias
#             Q_context[task] = (1 - learning_rate_context_td)*Q_context[task] + learning_rate_context_td*label_prob[task] # initialize tast set reward with task probability based on context

        # initialize reward of new tentative task set
#         Q_context[-1] = torch.max(Q_task[:-1])

#         retrieved_weight = mem_model.memories['params'][nearest_task]['weight']
#         retrieved_size = retrieved_weight.shape[1]
#         retrieved_bias = mem_model.memories['params'][nearest_task]['bias']
#         dyna_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
#         dyna_model.ensemble.bias.data = retrieved_bias

        print('train gate')
        while keep_training_gate:


#             if i < n_try_old:
            # select action that will give the highest expected reward
            action_ind = torch.where(Q_experts == torch.max(Q_experts))[0]
            best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
            action_list[i%feedback_interval] = best_action

            with torch.no_grad():
                preds = dyna_model(train_data['input'][i:(i+1), :, :])
                current_loss = criterion(preds[best_action].view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = current_loss.item()
            train_acc[i] = (preds[best_action].view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()


#             preds = dyna_model(train_data['input'][i:(i+1), :, :])
#             optimizer_dyna.zero_grad()
#             loss = criterion(preds[best_action].view(1, -1), train_data['target'][i].view(-1))
#             if train_old_expert:
#                 loss.backward()
#                 optimizer_dyna.step()
#             train_loss[i] = loss.item()
#             train_acc[i] = (preds[best_action].view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

#             print('best action: ' + str(int(best_action)))
            for expert_ind in range(dyna_model.num_experts):
                current_acc = (preds[expert_ind].view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
                acc_list[i%feedback_interval][expert_ind] = current_acc[0]
#                 acc[expert_ind] += current_acc[0]/feedback_interval
                if not (i+1) % feedback_interval:
                    for acc in acc_list:
                        Q_experts[expert_ind] = (1 - learning_rate_td)*Q_experts[expert_ind] + learning_rate_td*acc[expert_ind]
                        acc[expert_ind] = 0.0

            if train_old_expert:
                if not (i+1) % feedback_interval:
                    for j in range(feedback_interval):
                        ind = i - (feedback_interval - j - 1)
                        preds = dyna_model(train_data['input'][ind:(ind+1), :, :])
                        loss = criterion(preds[action_list[j]].view(1, -1), train_data['target'][ind].view(-1))
                        loss.backward()
                        optimizer_dyna.step()
                        optimizer_dyna.zero_grad()

#                 if expert_ind == best_action:
#                     train_acc[i] = acc
#             else:
#                 train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

#             if best_action == mem_model.n_tasks - 1:
#                 train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
#                 Q_task[-1] = (1 - learning_rate_td)*Q_task[-1] + learning_rate_td*train_acc[i]
#             else:
#                 state_dict = dyna_model.ensemble.state_dict()
#                 for task in range(mem_model.n_tasks-1):
#                     retrieved_weight = mem_model.memories['params'][task]['weight']
#                     retrieved_size = retrieved_weight.shape[1]
#                     dyna_model.ensemble.weight.data = torch.zeros_like(dyna_model.ensemble.weight.data)
#                     dyna_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
#                     dyna_model.ensemble.bias.data = mem_model.memories['params'][task]['bias']
#                     with torch.no_grad():
#                         pred = dyna_model(train_data['input'][i:(i+1), :, :])[1]
#                         acc = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
#                         Q_task[task] = (1 - learning_rate_td)*Q_task[task] + learning_rate_td*acc
#                     if task == best_action:
#                         train_acc[i] = acc
#                 if i == n_try_old - 1:
#                     print('reinstate task set ' + str(int(best_action)))
#                     retrieved_weight = mem_model.memories['params'][int(best_action)]['weight']
#                     retrieved_size = retrieved_weight.shape[1]
#                     dyna_model.ensemble.weight.data = torch.zeros_like(dyna_model.ensemble.weight.data)
#                     dyna_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
#                     dyna_model.ensemble.bias.data = mem_model.memories['params'][int(best_action)]['bias']
#                 else:
#                     dyna_model.ensemble.load_state_dict(state_dict)
            # store memories
#             mem_model.add_memory(train_data['input'][i])
#             mem_model.store_params({'weight':dyna_model.ensemble.weight.data.clone(), 'bias':dyna_model.ensemble.bias.data.clone()})

            i += 1

            if not i % print_every:
                print('DyaMoE {}/{} iterations complete'.format(i, n_train))

            if i == epochs_gate_check:
                perf_slope = torch.mean(train_acc[(i-acc_window):i]) - torch.mean(train_acc[:acc_window])
                perf = torch.mean(train_acc[(i-acc_window):i])
                print('acc slope: ' + str(perf_slope))
                print('acc: ' + str(perf))
                if  perf_slope < perf_slope_check and perf < perf_check:
                    keep_training_gate = False
                    print('ensemble performance is insufficient: recruit new expert')
                else:
                    train_old_expert = True
                    dyna_model.unfreeze_expert([i for i in range(dyna_model.num_experts)])
                    params = [
                        {'params':[expert[0].bias for expert in dyna_model.experts] + [expert[2].bias for expert in dyna_model.experts] + [expert[4].bias for expert in dyna_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        },
                        {'params':[expert[0].weight for expert in dyna_model.experts] + [expert[2].weight for expert in dyna_model.experts] + [expert[4].weight for expert in dyna_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        }
                    ]
                    optimizer_dyna = optim.Adam(params, lr=learning_rate_experts)
                    optimizer_dyna.zero_grad()
                    print('ensemble performance is sufficient: unfreeze experts')
            elif i >= n_train:
                keep_training_gate = False
                keep_training = False

        if keep_training:
            print('expert added at iteration {}'.format(str(i)))
            dyna_model.add_expert()

#             # copy best old expert's params to new expert
#             dyna_model.experts[-1].load_state_dict(dyna_model.experts[best_action].state_dict())

            params = [
                {'params':[param for expert in dyna_model.experts for i in [0, 2, 4] for param in expert[i].parameters()], 'lr':learning_rate_experts}
            ]
            optimizer_dyna = optim.Adam(params, lr=learning_rate_experts)
            criterion = torch.nn.NLLLoss()
            optimizer_dyna.zero_grad()

            print('train new expert')
            while keep_training:

#                 _ = trainEncoder(mem_model, train_data['input'][i], learning_rate = learning_rate_mem, margin = margin, replay_prob = rpe, use_compressed = use_compressed)

#                 nearest_task = int(predictNEM(mem_model, train_data['input'][i], use_compressed, k).topk(1)[1].squeeze().numpy())
#                 retrieved_weight = mem_model.memories['params'][nearest_task]['weight']
#                 retrieved_size = retrieved_weight.shape[1]
#                 retrieved_bias = mem_model.memories['params'][nearest_task]['bias']
#                 weight_temp = torch.zeros_like(dyna_model.ensemble.weight.data)
#                 weight_temp[:, :retrieved_size] = retrieved_weight
#                 dyna_model.ensemble.weight.data = mix_ratio*weight_temp + (1 - mix_ratio)*dyna_model.ensemble.weight.data
#                 dyna_model.ensemble.bias.data = mix_ratio*retrieved_bias + (1 - mix_ratio)*dyna_model.ensemble.bias.data

                with torch.no_grad():
                    pred = dyna_model(train_data['input'][i:(i+1), :, :])[-1]
                    current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
                train_loss[i] = current_loss.item()
                train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

                if not (i+1) % feedback_interval:
                    for j in range(feedback_interval):
                        ind = i - (feedback_interval - j - 1)
                        pred = dyna_model(train_data['input'][ind:(ind+1), :, :])[-1]
                        loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                        loss.backward()
                        optimizer_dyna.step()
                        optimizer_dyna.zero_grad()

#                 optimizer_dyna.zero_grad()

#                 pred = dyna_model(train_data['input'][i:(i+1), :, :])[-1]
#                 loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
#                 train_loss[i] = loss.item()
#                 train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

#                 loss.backward()
#                 optimizer_dyna.step()

                i += 1
                if not i % print_every:
                    print('DyaMoE {}/{} iterations complete'.format(i, n_train))

                if i >= n_train:
                    keep_training = False

    return train_loss, train_acc

def predict(dyna_model, input_tensor):
    with torch.no_grad():
        preds = dyna_model(input_tensor)
    return preds

def eval_dyna(model, input_tensor, target_tensor):
  n = input_tensor.shape[0]
  preds = torch.zeros((n, 2)).to(device)

  with torch.no_grad():
    expert_preds = model(input_tensor)

  for i in range(n):
    selected_expert = int(torch.floor(torch.rand(1)*model.num_experts))
    preds[i] = expert_preds[selected_expert][i].squeeze()

  acc = (preds.topk(1, dim=1)[1].squeeze() == target_tensor.squeeze()).float().mean()

  return acc


def trainTUDYNAMOE(dyna_model, train_data, dyna_hyper, feedback_interval=1):

    learning_rate_experts = dyna_hyper['learning_rate_experts']
    learning_rate_unfreeze = dyna_hyper['learning_rate_unfreeze']
    learning_rate_td = dyna_hyper['learning_rate_td'] # temporal difference learning rate
    perf_slope_check = dyna_hyper['perf_slope_check']
    perf_check = dyna_hyper['perf_check']
    epochs_gate_check = dyna_hyper['epochs_gate_check']
    acc_window = dyna_hyper['acc_window']
    mix_ratio = dyna_hyper['mix_ratio'] # number in [0, 1] specifying how much to mix old memory ensemble with current ensemble
    new_task_threshold = cre_hyper['new_task_threshold']

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for expert in dyna_model.experts for i in [0, 2, 4] for param in expert[i].parameters()], 'lr':learning_rate_experts}
    ]
    optimizer_dyna = optim.Adam(params, lr=learning_rate_experts)
    criterion = torch.nn.NLLLoss()
    optimizer_dyna.zero_grad()
    loss_list = [0.0]*feedback_interval

    # train_mode is one of 'expert', 'ensemble', or 'reinforcement'
    # start in 'expert' mode
    train_mode = 'expert'
    print('train expert')
    within_mode_idx = 0
    within_task_idx = 0

    keep_training = True

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    chance_acc = (train_data['target'] == torch.mode(train_data['target'], dim=0)[0]).float().mean()
    recent_acc = (torch.rand(acc_window).to(device) < chance_acc).float()
    best_acc = chance_acc

    for i in range(n_train):
        if train_mode == 'expert':

            with torch.no_grad():
                pred = dyna_model(train_data['input'][i:(i+1), :, :])[-1]
                current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = current_loss.item()
            train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
            recent_acc[:-1] = recent_acc[1:].clone()
            recent_acc[-1] = train_acc[i]

            if not (within_task_idx+1) % feedback_interval:
                for j in range(feedback_interval):
                    ind = i - (feedback_interval - j - 1)
                    pred = dyna_model(train_data['input'][ind:(ind+1), :, :])[-1]
                    loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                    loss.backward()
                    optimizer_dyna.step()
                    optimizer_dyna.zero_grad()

            within_mode_idx += 1
            within_task_idx += 1

            mean_acc = recent_acc.mean()
            if mean_acc > best_acc:
                best_acc = mean_acc
            elif best_acc - mean_acc > new_task_threshold:
                # we think a new task has been encountered
                dyna_model.freeze_expert(list(range(dyna_model.num_experts)))

                # when we think a new task is encountered, enter reinforcement mode
                train_mode = 'reinforcement'
                print('new task at iteration {}'.format(str(i)))
                print('train reinforcement')
                time.sleep(1.)
                within_mode_idx = 0
                within_task_idx = 0
                best_acc = chance_acc

              # initialize expected rewards for each expert
                Q_experts = torch.zeros(dyna_model.num_experts)
                acc_list = [torch.zeros(dyna_model.num_experts) for j in range(feedback_interval)]
                action_list = [0]*feedback_interval

                train_old_expert = False

        elif train_mode == 'reinforcement':

            action_ind = torch.where(Q_experts == torch.max(Q_experts))[0]
            best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
            action_list[within_task_idx%feedback_interval] = best_action

            with torch.no_grad():
                preds = dyna_model(train_data['input'][i:(i+1), :, :])
                current_loss = criterion(preds[best_action].view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = current_loss.item()
            train_acc[i] = (preds[best_action].view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

            for expert_ind in range(dyna_model.num_experts):
                current_acc = (preds[expert_ind].view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
                acc_list[within_task_idx%feedback_interval][expert_ind] = current_acc[0]
#                 acc[expert_ind] += current_acc[0]/feedback_interval
                if not (within_task_idx+1) % feedback_interval:
                    for acc in acc_list:
                        Q_experts[expert_ind] = (1 - learning_rate_td)*Q_experts[expert_ind] + learning_rate_td*acc[expert_ind]
                        acc[expert_ind] = 0.0

            if train_old_expert:
                if not (within_task_idx+1) % feedback_interval:
                    for j in range(feedback_interval):
                        ind = i - (feedback_interval - j - 1)
                        preds = dyna_model(train_data['input'][ind:(ind+1), :, :])
                        loss = criterion(preds[action_list[j]].view(1, -1), train_data['target'][ind].view(-1))
                        loss.backward()
                        optimizer_dyna.step()
                        optimizer_dyna.zero_grad()

            within_task_idx += 1

            if within_mode_idx == (epochs_gate_check-1):
                perf_slope = torch.mean(train_acc[(i-acc_window):i]) - torch.mean(train_acc[:acc_window])
                perf = torch.mean(train_acc[(i-acc_window):i])
                print('acc slope: ' + str(perf_slope))
                print('acc: ' + str(perf))
                if perf < perf_check:
                    print('performance is insufficient: recruit new expert at iteration {}'.format(str(i)))
                    time.sleep(1.)
                    train_mode = 'expert'
                    within_mode_idx = 0
                    dyna_model.add_expert()
                    params = [
                        {'params':[param for expert in dyna_model.experts for i in [0, 2, 4] for param in expert[i].parameters()], 'lr':learning_rate_experts}
                        ]
                    optimizer_dyna = optim.Adam(params, lr=learning_rate_experts)
                    criterion = torch.nn.NLLLoss()
                    optimizer_dyna.zero_grad()
                else:
                    within_mode_idx += 1
                    train_old_expert = True
                    dyna_model.unfreeze_expert([i for i in range(dyna_model.num_experts)])
                    params = [
                        {'params':[expert[0].bias for expert in dyna_model.experts] + [expert[2].bias for expert in dyna_model.experts] + [expert[4].bias for expert in dyna_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        },
                        {'params':[expert[0].weight for expert in dyna_model.experts] + [expert[2].weight for expert in dyna_model.experts] + [expert[4].weight for expert in dyna_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        }
                    ]
                    optimizer_dyna = optim.Adam(params)
                    optimizer_dyna.zero_grad()
                    print('ensemble performance is sufficient: unfreeze experts')
                    time.sleep(1.)
            elif within_mode_idx > (epochs_gate_check - 1):
                mean_acc = recent_acc.mean()
                if mean_acc > best_acc:
                    best_acc = mean_acc
                elif best_acc - mean_acc > new_task_threshold:
                    # we think a new task has been encountered
                    dyna_model.freeze_expert([dyna_model.num_experts])
                    mem_model.store_params({'weight':dyna_model.ensemble.weight.data.clone(), 'bias':dyna_model.ensemble.bias.data.clone()})

                    # when we think a new task is encountered, enter reinforcement mode
                    train_mode = 'reinforcement'
                    within_mode_idx = 0
                    within_task_idx = 0
                    best_acc = chance_acc

                    # initialize expected rewards for each expert
                    Q_experts = torch.zeros(dyna_model.num_experts)
                    acc_list = [torch.zeros(dyna_model.num_experts) for j in range(feedback_interval)]
                    action_list = [0]*feedback_interval

                    train_old_expert = False
            else:
                within_mode_idx += 1

        if not i % print_every:
            print('DynaMoE {}/{} iterations complete'.format(i, n_train))

    return train_loss, train_acc

def trainRE(cre_model, mem_model, train_data, cre_hyper, mem_hyper):

    learning_rate_experts = cre_hyper['learning_rate_experts']
    learning_rate_ensemble = cre_hyper['learning_rate_ensemble']
    learning_rate_unfreeze = cre_hyper['learning_rate_unfreeze']
    learning_rate_td = cre_hyper['learning_rate_td']
    perf_slope_check = cre_hyper['perf_slope_check']
    perf_check = cre_hyper['perf_check']
    epochs_ensemble_check = cre_hyper['epochs_ensemble_check']
    acc_window = cre_hyper['acc_window']
#     l2_lambda = cre_hyper['l2_lambda']
    mix_ratio = cre_hyper['mix_ratio'] # number in [0, 1] specifying how much to mix old memory ensemble with current ensemble
    n_try_old = cre_hyper['n_try_old']

    learning_rate_mem = mem_hyper['learning_rate']
    margin = mem_hyper['margin']
    decoder_epochs = mem_hyper['epochs']
    replay_prob_encoder = mem_hyper['replay_prob_encoder']
    replay_prob_decoder = mem_hyper['replay_prob_decoder']
    k = mem_hyper['k']

    if mem_model.n_tasks > 2:
        use_compressed = True
        rpe = replay_prob_encoder
    else:
        use_compressed = False
        rpe = 1.


    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in cre_model.ensemble.parameters()], 'lr':learning_rate_ensemble},
        {'params':[param for expert in cre_model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
    ]
    optimizer_cre = optim.Adam(params, lr=learning_rate_experts)
    criterion = torch.nn.NLLLoss()

    keep_training = True

    # if first expert, then train the expert and the ensemble network
    if cre_model.num_experts == 1 and not cre_model.frozen[-1]:
        print('train first expert')

        train_loss = torch.zeros(n_train)
        train_acc = torch.zeros(n_train)

        for i in range(n_train):

            optimizer_cre.zero_grad()

            pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
            loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = loss.item()
            train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

            loss.backward()
            optimizer_cre.step()

            # store memories
            mem_model.add_memory(train_data['input'][i])

            if not i % print_every:
                print('CRE {}/{} iterations complete'.format(i, n_train))

        cre_model.freeze_expert([0])

        mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

    # try tuning the ensemble network
    else:
        cre_model.freeze_expert([i for i in range(cre_model.num_experts)])

        # initialize expected rewards for each task set
        Q = torch.zeros(mem_model.n_tasks)

        train_loss = torch.zeros(n_train)
        train_acc = torch.zeros(n_train)

        keep_training_ensemble = True

        i = 0

        # initialize ensemble parameters with randomly sampled prior task set
        task = int(torch.randint(high=mem_model.n_tasks-1, size=(1,)))
        retrieved_weight = mem_model.memories['params'][task]['weight']
        retrieved_size = retrieved_weight.shape[1]
        cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data)
        cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
        cre_model.ensemble.bias.data = mem_model.memories['params'][task]['bias']

        # initialize all action rewards to chance accuracy
        Q[:] = 1./cre_model.num_classes

        print('train ensemble')
        while keep_training_ensemble:

            optimizer_cre.zero_grad()

            pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
            loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
            train_loss[i] = loss.item()

            loss.backward()
            optimizer_cre.step()

            if i < n_try_old:
                # select action that will give the highest expected reward
                action_ind = torch.where(Q == torch.max(Q))[0]
                best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
                print('best action: ' + str(int(best_action)))
                if best_action == mem_model.n_tasks - 1:
                    train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
                    Q[-1] = (1 - learning_rate_td)*Q[-1] + learning_rate_td*train_acc[i]
                else:
                    state_dict = cre_model.ensemble.state_dict()
                    for task in range(mem_model.n_tasks-1):
                        retrieved_weight = mem_model.memories['params'][task]['weight']
                        retrieved_size = retrieved_weight.shape[1]
                        cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data)
                        cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                        cre_model.ensemble.bias.data = mem_model.memories['params'][task]['bias']
                        with torch.no_grad():
                            pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                            acc = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
                            Q[task] = (1 - learning_rate_td)*Q[task] + learning_rate_td*acc
                        if task == best_action:
                            train_acc[i] = acc
                    if i == n_try_old - 1:
                        print('reinstate task set ' + str(int(best_action)))
                        retrieved_weight = mem_model.memories['params'][int(best_action)]['weight']
                        retrieved_size = retrieved_weight.shape[1]
                        cre_model.ensemble.weight.data = torch.zeros_like(cre_model.ensemble.weight.data)
                        cre_model.ensemble.weight.data[:, :retrieved_size] = retrieved_weight
                        cre_model.ensemble.bias.data = mem_model.memories['params'][int(best_action)]['bias']
                    else:
                        cre_model.ensemble.load_state_dict(state_dict)
            else:
                train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

            mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

            i += 1

            if not i % print_every:
                print('CRE {}/{} iterations complete'.format(i, n_train))

            if i == epochs_ensemble_check:
                perf_slope = torch.mean(train_acc[(i-acc_window):i]) - torch.mean(train_acc[:acc_window])
                perf = torch.mean(train_acc[(i-acc_window):i])
                print('acc slope: ' + str(perf_slope))
                print('acc: ' + str(perf))
                if  perf_slope < perf_slope_check and perf < perf_check:
                    keep_training_ensemble = False
                    print('ensemble performance is insufficient: recruit new expert')
                else:
                    cre_model.unfreeze_expert([i for i in range(cre_model.num_experts)])
                    params = [
                        {'params':[param for param in cre_model.ensemble.parameters()],
                         'weight_decay':0.,
                         'lr':learning_rate_ensemble
                        },
                        {'params':[expert[0].bias for expert in cre_model.experts] + [expert[3].bias for expert in cre_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        },
                        {'params':[expert[0].weight for expert in cre_model.experts] + [expert[3].weight for expert in cre_model.experts],
                         'weight_decay':0.,
                         'lr':learning_rate_unfreeze
                        }
                    ]
                    optimizer_cre = optim.Adam(params, lr=learning_rate_experts)
                    print('ensemble performance is sufficient: unfreeze experts')
            elif i >= n_train:
                keep_training_ensemble = False
                keep_training = False

        if keep_training:
            print('expert added at iteration {}'.format(str(i)))
            cre_model.add_expert()

            params = [
                {'params':[param for param in cre_model.ensemble.parameters()], 'lr':learning_rate_ensemble},
                {'params':[param for expert in cre_model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
            ]
            optimizer_cre = optim.Adam(params, lr=learning_rate_experts)
            criterion = torch.nn.NLLLoss()

            print('train new expert')
            while keep_training:

                optimizer_cre.zero_grad()

                pred = cre_model(train_data['input'][i:(i+1), :, :])[1]
                loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
                train_loss[i] = loss.item()
                train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

                loss.backward()
                optimizer_cre.step()

                mem_model.store_params({'weight':cre_model.ensemble.weight.data.clone(), 'bias':cre_model.ensemble.bias.data.clone()})

                i += 1
                if not i % print_every:
                    print('CRE {}/{} iterations complete'.format(i, n_train))

                if i >= n_train:
                    keep_training = False

    return train_loss, train_acc


def myTripletLoss(query, positive_neighbor, negative_neighbor, margin):
    loss = torch.maximum(torch.dot(query, negative_neighbor) - torch.dot(query, positive_neighbor) + torch.tensor([margin], device=device), torch.tensor([0], device=device))
    return loss

def myTripletLoss2(query, positive_neighbor, negative_neighbor, margin):
    loss = torch.maximum(torch.linalg.norm(query - positive_neighbor) - torch.linalg.norm(query - negative_neighbor) + torch.tensor([margin]), torch.tensor([0]))
    return loss

def trainEncoder(model, input_tensor, learning_rate = 1e-3, margin = 0.1, replay_prob = 1., use_compressed = False):
#     n_train = input_tensor.shape[0]
#     n_val = val_data['input'].shape[0]

    optimizer = optim.Adam(model.memory_encoder.parameters(), lr=learning_rate)

    optimizer.zero_grad()

    if use_compressed:
        # previous task memories are already compressed
        old_task_memories = torch.cat([model.memories['compressed'][int(task*model.max_memories):int(torch.tensor([model.n_memories[task], model.max_memories]).min().numpy() + task*model.max_memories), :] for task in range(model.n_tasks-1)], dim=0)

        # current task memories are not compressed yet, since AE hasn't been trained
        cur_task_memories = model.memories['input'][int((model.n_tasks-1)*model.max_memories):int(torch.tensor([model.n_memories[model.n_tasks-1], model.max_memories]).min().numpy() + (model.n_tasks-1)*model.max_memories), :]
        cur_task_memories = model.memory_encoder(cur_task_memories)

        memory_tensor = torch.cat([old_task_memories, cur_task_memories], dim=0)
    else:
        memory_tensor = torch.cat([model.memories['input'][int(task*model.max_memories):int(torch.tensor([model.n_memories[task], model.max_memories]).min().numpy() + task*model.max_memories), :] for task in range(model.n_tasks)], dim=0)
        memory_tensor = model.memory_encoder(memory_tensor)

    n_mem = memory_tensor.shape[0]
    memory_tensor = F.normalize(memory_tensor.view(n_mem, -1), dim=1)
    memory_task = torch.cat([model.memories['task'][int(task*model.max_memories):int(torch.tensor([model.n_memories[task], model.max_memories]).min().numpy()  + task*model.max_memories)] for task in range(model.n_tasks)])

#     memory_inds = []
#     for task in range(model.n_tasks):
#         memory_inds += [i + task*model.max_memories for i in range(np.min([model.n_memories[task], model.max_memories]))]

    # ///compute loss with new data point as the query///
    query_tensor = model.memory_encoder(input_tensor)
    query_tensor = F.normalize(query_tensor.view(1, -1), dim=1)
    query_task = model.n_tasks - 1

    cosine_sim = torch.mm(query_tensor, memory_tensor.T)

    # sort cosine similarities along rows
    sorted_sim, sorted_ind = torch.sort(cosine_sim, descending=True)

    loss = 0

    # find positive and negative neighbors
    positive_idx = -1
    negative_idx = -1
    keep_searching = True
    idx = 0
    while keep_searching:
        idx_sort = sorted_ind[:, idx]
        if positive_idx < 0:
            if memory_task[idx_sort] == query_task: #///////
                positive_idx = idx_sort
        if negative_idx < 0:
            if memory_task[idx_sort] != query_task:
                negative_idx = idx_sort
        if positive_idx*negative_idx > 0:
            keep_searching = False
        idx += 1
        if idx == n_mem:
            keep_searching = False

    loss += myTripletLoss(query_tensor.squeeze(), memory_tensor[positive_idx, :].squeeze(), memory_tensor[negative_idx, :].squeeze(), margin)
    # ///compute loss with new data point as the query///

    if torch.rand(1) < replay_prob:
        # ///compute loss with negative neighbor as the query///
        query_tensor = memory_tensor[negative_idx, :]

        if use_compressed:
            # decompress memory
            with torch.no_grad():
                query_tensor = model.memory_decoder(query_tensor)

            # recompress memory through encoder
            query_tensor = model.memory_encoder(query_tensor)

        query_task = int(memory_task[negative_idx].cpu().numpy())

        cosine_sim = torch.mm(query_tensor, memory_tensor.T)

        # make similarity of query point with itself < -1 so that it will be last in the sort
        cosine_sim[:, negative_idx] = -2.

        # sort cosine similarities along rows
        sorted_sim, sorted_ind = torch.sort(cosine_sim, descending=True)

        # find positive and negative neighbors
        positive_idx = -1
        negative_idx = -1
        keep_searching = True
        idx = 0
        while keep_searching:
            idx_sort = sorted_ind[:, idx]
            if positive_idx < 0:
                if memory_task[idx_sort] == query_task: #///////
                    positive_idx = idx_sort
            if negative_idx < 0:
                if memory_task[idx_sort] != query_task:
                    negative_idx = idx_sort
            if positive_idx*negative_idx > 0:
                keep_searching = False
            idx += 1
            if idx == n_mem:
                keep_searching = False
        loss += myTripletLoss(query_tensor.squeeze(), memory_tensor[positive_idx, :].squeeze(), memory_tensor[negative_idx, :].squeeze(), margin)
        # ///compute loss with negative neighbor as the query///

    loss.backward()
    optimizer.step()

    return loss

def trainDecoder(model, encoded_tensor, target_tensor, learning_rate = 1e-3):

    optimizer = optim.Adam(model.memory_decoder.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    optimizer.zero_grad()
    decoded_tensor = model.memory_decoder(encoded_tensor)
    loss = criterion(decoded_tensor, target_tensor)
    loss.backward()
    optimizer.step()

    return loss

def predictNEM(model, input_tensor, use_compressed = False, k = 1):
    n = input_tensor.shape[0]
    # k is the number of NN for predicting
    with torch.no_grad():

        if use_compressed:
            # previous task memories are already compressed
            old_task_memories = torch.cat([model.memories['compressed'][int(task*model.max_memories):int(torch.tensor([model.n_memories[task], model.max_memories]).min().numpy() + task*model.max_memories), :] for task in range(model.n_tasks-1)], dim=0)

            # current task memories are not compressed yet, since AE hasn't been trained
            cur_task_memories = model.memories['input'][int((model.n_tasks-1)*model.max_memories):int(torch.tensor([model.n_memories[model.n_tasks-1], model.max_memories]).min().numpy() + (model.n_tasks-1)*model.max_memories), :]
            cur_task_memories = model.memory_encoder(cur_task_memories)

            memory_tensor = torch.cat([old_task_memories, cur_task_memories], dim=0)

        else:
            memory_tensor = torch.cat([model.memories['input'][int(task*model.max_memories):int(torch.tensor([model.n_memories[task], model.max_memories]).min().numpy() + task*model.max_memories), :] for task in range(model.n_tasks)], dim=0)
            memory_tensor = model.memory_encoder(memory_tensor)

        n_mem = memory_tensor.shape[0]
        memory_tensor = F.normalize(memory_tensor.view(n_mem, -1), dim=1)
        memory_task = torch.cat([model.memories['task'][int(task*model.max_memories):int(torch.tensor([model.n_memories[task], model.max_memories]).min().numpy()  + task*model.max_memories)] for task in range(model.n_tasks)])

        query_tensor = model.memory_encoder(input_tensor)
        query_tensor = F.normalize(query_tensor.view(n, -1), dim=1)

        cosine_sim = torch.mm(query_tensor, memory_tensor.T)

        # sort cosine similarities along rows
        sorted_sim, sorted_ind = torch.sort(cosine_sim, descending=True)

        label_prob = torch.zeros((n, model.n_tasks))

        for i in range(n):
            knn_labels = torch.index_select(memory_task, 0, sorted_ind[i, :k])
            for j in range(k):
                label_prob[i, int(knn_labels[j].cpu().numpy())] += 1/k

    return label_prob

def computeAccuracy(label_prob, ground_truth):
    return (label_prob.argmax(dim=1) == ground_truth).sum()/ground_truth.shape[0]

def eval_nem(mem_model, input_tensor, task_label_tensor, num_tasks, use_compressed=False):
    n = input_tensor.shape[0]
    task_label_preds = torch.zeros(n)
    unique_task_labels = torch.tensor(list(range(num_tasks)))

    for i in range(n):
        task_label_preds[i] = predictNEM(mem_model, input_tensor[i], use_compressed, 1).squeeze().topk(1)[1].squeeze()

    task_label_preds = task_label_preds.to(device)
    task_label_acc = (task_label_preds == task_label_tensor.squeeze()).float().mean()
    task_label_confusion = confusionMatrix(task_label_tensor.squeeze(), task_label_preds, num_tasks, normalize=False)

    return task_label_acc, task_label_confusion

def trainSimple(model, train_data, hyper, feedback_interval=1, cre=False):

    learning_rate_experts = hyper['learning_rate_experts']
    learning_rate_ensemble = hyper['learning_rate_ensemble']

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in model.ensemble.parameters()], 'lr':learning_rate_ensemble},
        {'params':[param for expert in model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
    ]
    optimizer = optim.Adam(params)
    criterion = torch.nn.NLLLoss()
    loss_list = [0.0]*feedback_interval
    optimizer.zero_grad()

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    for i in range(n_train):
        with torch.no_grad():
            pred = model(train_data['input'][i:(i+1), :, :])
            if cre:
              pred = pred[1]
            current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
        train_loss[i] = current_loss.item()
        train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

        if not (i+1) % feedback_interval:
            for j in range(feedback_interval):
                ind = i - (feedback_interval - j - 1)
                pred = model(train_data['input'][ind:(ind+1), :, :])
                if cre:
                  pred = pred[1]
                loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


#         optimizer.zero_grad()
#         pred = model(train_data['input'][i:(i+1), :, :])
#         loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
#         train_loss[i] = loss.item()
#         train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
#         loss.backward()
#         optimizer.step()
        if not i % print_every:
            print('Scratch/FineTune {}/{} iterations complete'.format(i, n_train))

    return train_loss, train_acc


def trainMultihead(model, num_heads, train_data, hyper, feedback_interval=1, cre=False):

    learning_rate_experts = hyper['learning_rate_experts']
    learning_rate_ensemble = hyper['learning_rate_ensemble']

    head_weights = [model.ensemble.weight.data.clone() for i in range(num_heads)]
    head_biases = [model.ensemble.bias.data.clone() for i in range(num_heads)]

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in model.ensemble.parameters()], 'lr':learning_rate_ensemble},
        {'params':[param for expert in model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
    ]
    optimizer = optim.Adam(params)
    criterion = torch.nn.NLLLoss()
    loss_list = [0.0]*feedback_interval
    optimizer.zero_grad()

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    for i in range(n_train):
      head_id = int(train_data['label'][i])
      model.ensemble.weight.data[:] = head_weights[head_id]
      model.ensemble.bias.data[:] = head_biases[head_id]
      with torch.no_grad():
          pred = model(train_data['input'][i:(i+1), :, :])
          if cre:
            pred = pred[1]
          current_loss = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
      train_loss[i] = current_loss.item()
      train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

      if not (i+1) % feedback_interval:
          for j in range(feedback_interval):
              ind = i - (feedback_interval - j - 1)
              pred = model(train_data['input'][ind:(ind+1), :, :])
              if cre:
                pred = pred[1]
              loss = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
              loss.backward()
              optimizer.step()
              optimizer.zero_grad()
          head_weights[head_id][:] = model.ensemble.weight.data.clone()
          head_biases[head_id][:] = model.ensemble.bias.data.clone()


      if not i % print_every:
          print('Scratch/FineTune {}/{} iterations complete'.format(i, n_train))

    return train_loss, train_acc, head_weights, head_biases

def eval_multihead(model, input_tensor, target_tensor):
  n = input_tensor.shape[0]

  with torch.no_grad():
    preds = model(input_tensor).view(n, -1)

  acc = (preds.topk(1, dim=1)[1].squeeze() == target_tensor.squeeze()).float().mean()

  return acc


def trainEWC(model, train_data, hyper, exclude_tasks=[], feedback_interval=1):

    learning_rate_experts = hyper['learning_rate_experts']
    learning_rate_ensemble = hyper['learning_rate_ensemble']
    lambda_ewc = hyper['lambda_ewc']

    n_train = train_data['input'].shape[0]

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in model.ensemble.parameters()], 'lr':learning_rate_ensemble},
        {'params':[param for expert in model.experts for i in [0, 3] for param in expert[i].parameters()], 'lr':learning_rate_experts}
    ]
    optimizer = optim.Adam(params)
    criterion = torch.nn.NLLLoss()
    loss_list = [0.0]*feedback_interval
    optimizer.zero_grad()

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    for i in range(n_train):
        with torch.no_grad():
            pred = model(train_data['input'][i:(i+1), :, :])
            loss_nll = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
            loss_ewc = regEWC(model, exclude_tasks)
            current_loss = loss_nll + lambda_ewc*loss_ewc
        train_loss[i] = current_loss.item()
        train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()

        if not (i+1) % feedback_interval:
            for j in range(feedback_interval):
                ind = i - (feedback_interval - j - 1)
                pred = model(train_data['input'][ind:(ind+1), :, :])
                loss_nll = criterion(pred.view(1, -1), train_data['target'][ind].view(-1))
                loss_ewc = regEWC(model, exclude_tasks)
                loss = loss_nll + lambda_ewc*loss_ewc
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

#         optimizer.zero_grad()
#         pred = model(train_data['input'][i:(i+1), :, :])
#         loss_nll = criterion(pred.view(1, -1), train_data['target'][i].view(-1))
#         loss_ewc = regEWC(model, exclude_tasks)
#         loss = loss_nll + lambda_ewc*loss_ewc
#         train_loss[i] = loss.item()
#         train_acc[i] = (pred.view(-1).topk(1)[1] == train_data['target'][i].view(-1)).float()
#         loss.backward()
#         optimizer.step()
        if not i % print_every:
            print('EWC {}/{} iterations complete'.format(i, n_train))

    return train_loss, train_acc

def regEWC(model, exclude_tasks):
#     cur_params = [param for param in model.parameters()]
    loss_ewc = 0
    for task_ind, task_name in enumerate(model.task_params.keys()):
        if task_name not in exclude_tasks:
            for layer_ind, param in enumerate(model.parameters()):
                loss_ewc += 0.5*(model.F[task_name][layer_ind]*(param - model.task_params[task_name][layer_ind])**2).squeeze().sum()
    return loss_ewc

def eval_ewc(model, input_tensor, target_tensor):
  n = input_tensor.shape[0]

  with torch.no_grad():
    preds = model(input_tensor).view(n, -1)

  acc = (preds.topk(1, dim=1)[1].squeeze() == target_tensor.squeeze()).float().mean()

  return acc
