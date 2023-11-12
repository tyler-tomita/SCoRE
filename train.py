import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from IPython import display
from time import sleep
import copy
import time
import math
from scipy import ndimage as ndi
from PIL import Image
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plot_markers = ['o', 's', '*', 'D', '^']

def trainDynaMoE(model, feature_extractor, dataloader, optimizer_class, hyper, task_start_inds = []):

    learning_rate_experts = hyper['learning_rate_experts']
    wd = hyper['weight_decay']
    learning_rate_td = hyper['learning_rate_td']
    perf_check_context_recognizer = hyper['perf_check_context_recognizer']
    acc_window = hyper['acc_window']
    n_try_old = hyper['n_try_old']
    new_task_threshold = hyper['new_task_threshold']
    check_new_task_acc = hyper['check_new_task_acc']
    amsgrad = hyper['amsgrad']
    max_experts = hyper['max_experts']

    n_train = len(dataloader.dataset)

    print_every = math.floor(n_train/10)

    # train_mode is one of 'expert' or 'context_recognizer'
    # start in 'expert' mode
    train_mode = 'expert'
    # print('train expert')
    within_mode_idx = 0
    within_task_idx = 0
    task_number = 0

    model.add_expert()
    model.task2expert.append(0)

    # set training parameters
    params = [
        {'params':[param for expert in model.experts for param in expert.parameters()],
         'lr':learning_rate_experts,
         'weight_decay':wd}
    ]
    optimizer = optimizer_class(params, amsgrad=amsgrad)
    optimizer.zero_grad()
    criterion = torch.nn.NLLLoss()

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    # compute chance accuracy and initialize recent_acc and best_acc with chance
    all_labels =torch.tensor([])
    for _, _, labels, _ in dataloader:
        all_labels = torch.cat([all_labels, labels])
    chance_acc = (all_labels.mode()[0] == all_labels).float().mean()
    # recent_acc = torch.zeros(100) + chance_acc
    # recent_acc = torch.zeros(100)
    # best_acc = chance_acc
    best_acc = 0.

    i = 0

    for inputs, _, labels, _ in dataloader:
        inputs = inputs.to(device)
        if feature_extractor:
            with torch.no_grad():
                inputs = feature_extractor(inputs)
        labels = labels.to(device)

        batch_size = len(labels)

        for ii in range(batch_size):
            if train_mode == 'expert':

                # train the expert
                train_loss[i], train_acc[i] = trainDynaExpert(model, inputs[ii:(ii+1)], labels[ii], optimizer, criterion)

                # compute mean of the past `acc_window` trials
                # recent_acc[:-1] = recent_acc[1:].clone()
                # recent_acc[-1] = train_acc[(i+1-acc_window):(i+1)].mean()
                # mean_acc = recent_acc[-1]
                mean_acc = train_acc[(i+1-acc_window):(i+1)].mean()

                # if the current average accuracy has decreased from
                # best accuracy by `new_task_threshold`, and the best_accuracy is
                # high enough (> `check_new_task_acc`), then predict a task-change
                if (i+1 in task_start_inds) or (((best_acc - mean_acc > new_task_threshold) and (best_acc > check_new_task_acc)) and not task_start_inds):
                    # we think a new task has been encountered
                    print(f'add task at trial {i}')
                    print(f'(mean accuracy = {mean_acc})')
                    print()
                    time.sleep(1.)
                    model.freeze_expert(list(range(model.num_experts)))

                    train_mode = 'context_recognizer'

                    # initialize RL reward for each task
                    Q_expert = torch.zeros(model.num_experts)

                    # update tracked variables
                    within_mode_idx = 0
                    within_task_idx = 0
                    # best_acc = chance_acc
                    # recent_acc[:] = chance_acc
                    task_number += 1
                else:
                    within_mode_idx += 1
                    within_task_idx += 1

            elif train_mode == 'context_recognizer':
                # select action that will give the highest expected reward
                action_ind = torch.where(Q_expert == torch.max(Q_expert))[0]
                best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]

                # get predictions from all experts
                model.eval()
                with torch.no_grad():
                    all_preds = model(inputs[ii:(ii+1)])
                model.train()

                # compare predictions to target label and update the rewards for each expert
                all_accs= torch.zeros(model.num_experts, dtype=torch.long)
                for expert_idx in range(model.num_experts):
                    preds = all_preds[expert_idx]
                    current_loss = criterion(preds.view(1, -1), labels[ii].view(-1))
                    current_acc = (preds.view(-1).topk(1)[1] == labels[ii].view(-1)).float().mean()
                    all_accs[expert_idx] = current_acc
                    Q_expert[expert_idx] = (1 - learning_rate_td)*Q_expert[expert_idx] + learning_rate_td*current_acc.item()
                    # store the loss for the ensembler that was actually employed
                    if expert_idx == best_action:
                        train_loss[i] = current_loss.clone().item()
                        train_acc[i] = current_acc.clone()

                mean_acc = train_acc[(i+1-acc_window):(i+1)].mean()

                if within_mode_idx == (n_try_old - 1):
                    # print('reinstate task set ' + str(int(best_action)))

                    # select the best Ensembler for the remainder of the new task context or create new one
                    action_ind = torch.where(Q_expert == torch.max(Q_expert))[0]
                    best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))].item()

                    # worst accuracy for current task is the average accuracy during the context recognizer phase
                    worst_acc = train_acc[(i+1-n_try_old):(i+1)].mean()

                    if (worst_acc < perf_check_context_recognizer) and (model.num_experts < max_experts):
                        print(f'add expert at trial {i}')
                        print(f'(mean context recognizer accuracy = {worst_acc})')
                        # print(recent_acc)
                        print()
                        time.sleep(1.)
                        model.add_expert()
                        best_action = model.num_experts - 1
                    else:
                        print(f'Reuse expert {best_action}')
                        print()
                        time.sleep(1.)
                        # model.unfreeze_ensembler([best_action])

                    model.task2expert.append(best_action)

                    # switch to ensemble mode
                    train_mode = 'expert'
                    within_mode_idx = 0

                    params = [
                        {'params':[param for expert in model.experts for param in expert.parameters()],
                         'lr':learning_rate_experts,
                         'weight_decay':wd}
                    ]
                    optimizer = optimizer_class(params, amsgrad=amsgrad)
                    optimizer.zero_grad()

                    model.unfreeze_expert([model.task2expert[-1]])

                else:
                    within_mode_idx += 1

                within_task_idx += 1

            # update the running best accuracy if it has increased
            if mean_acc > best_acc:
                best_acc = mean_acc

            i += 1

    return train_loss, train_acc

def trainDynaExpert(model, inputs, labels, optimizer, criterion, task_idx=-1):

    all_preds = model(inputs)
    preds = all_preds[model.task2expert[task_idx]]

    current_loss = criterion(preds.view(1, -1), labels.view(-1))
    train_loss = current_loss.item()
    train_acc = (preds.view(-1).topk(1)[1] == labels.view(-1)).float()

    current_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_loss, train_acc

def trainSCORE(score_model, mem_model, feature_extractor, dataloader, optimizer, score_hyper, mem_hyper, use_compressed=False, expert_decoder=None, task_start_inds = []):

    learning_rate_experts = score_hyper['learning_rate_experts']
    learning_rate_ensembler = score_hyper['learning_rate_ensembler']
    learning_rate_unfreeze = score_hyper['learning_rate_unfreeze']
    wd = score_hyper['weight_decay']
    learning_rate_td = score_hyper['learning_rate_td']
    learning_rate_context_td = score_hyper['learning_rate_context_td']
    context_reward_weight = score_hyper['context_reward_weight']
    perf_slope_check = score_hyper['perf_slope_check']
    perf_check_context_recognizer = score_hyper['perf_check_context_recognizer']
    perf_check_ensembler = score_hyper['perf_check_ensembler']
    epochs_ensembler_check = score_hyper['epochs_ensembler_check']
    acc_window = score_hyper['acc_window']
    # mix_ratio = score_hyper['mix_ratio'] # number in [0, 1] specifying how much to mix old memory ensemble with current ensemble
    n_try_old = score_hyper['n_try_old']
    new_task_threshold = score_hyper['new_task_threshold']
    check_new_task_acc = score_hyper['check_new_task_acc']
    replay_after_new = score_hyper['replay_after_new']
    amsgrad = score_hyper['amsgrad']
    lambda_autoencoder = score_hyper['lambda_autoencoder']
    reuse_ensembler_threshold = score_hyper['reuse_ensembler_threshold']
    max_experts = score_hyper['max_experts']

    learning_rate_mem = mem_hyper['learning_rate']
    margin = mem_hyper['margin']
    decoder_epochs = mem_hyper['epochs']
    replay_prob_encoder = mem_hyper['replay_prob_encoder']
    replay_prob_decoder = mem_hyper['replay_prob_decoder']
    k = mem_hyper['k']
    expert_replay_prob = mem_hyper['expert_replay_prob']

    rpe = replay_prob_encoder

    replay_old_task = False

    n_train = len(dataloader.dataset)

    print_every = math.floor(n_train/10)

    # train_mode is one of 'expert', 'ensembler', or 'context_recognizer'
    # start in 'expert' mode
    train_mode = 'expert'
    # print('train expert')
    within_mode_idx = 0
    within_task_idx = 0
    task_number = 0

    mem_model.add_task()
    # add first Expert and Ensembler
    score_model.add_expert()
    score_model.add_ensembler()
    score_model.task2ensembler.append(0)

    # set training parameters
    if expert_decoder:
        params = [
            {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
            'lr':learning_rate_experts,
            'weight_decay':wd},
            {'params':[param for expert in score_model.experts for param in expert.parameters()],
            'lr':learning_rate_experts,
            'weight_decay':wd},
            {'params':[param for param in expert_decoder.parameters()],
            'lr':learning_rate_experts,
            'weight_decay':wd}
        ]
        criterion_autoencoder = nn.MSELoss(reduction='none')

    else:
        params = [
            {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
            'lr':learning_rate_experts,
            'weight_decay':wd},
            {'params':[param for expert in score_model.experts for param in expert.parameters()],
            'lr':learning_rate_experts,
            'weight_decay':wd}
        ]
        criterion_autoencoder=None
        lambda_autoencoder = 0.

    optimizer_score = optimizer(params, amsgrad=amsgrad)
    optimizer_score.zero_grad()
    criterion = torch.nn.NLLLoss()

    loss_task_id = {}
    loss_task_id[mem_model.n_tasks-1] = []

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    # compute chance accuracy and initialize recent_acc and best_acc with chance
    all_labels =torch.tensor([])
    for _, _, labels, _ in dataloader:
        all_labels = torch.cat([all_labels, labels])
    chance_acc = (all_labels.mode()[0] == all_labels).float().mean()
    # recent_acc = torch.zeros(100) + chance_acc
    # recent_acc = torch.zeros(100)
    # best_acc = chance_acc
    best_acc = 0.

    i = 0

    for inputs, _, labels, _ in dataloader:
        inputs = inputs.to(device)
        if feature_extractor:
            with torch.no_grad():
                inputs = feature_extractor(inputs)
        labels = labels.to(device)

        batch_size = len(labels)

        for ii in range(batch_size):
            # if i < 384:
            #     print({})
            #     print(f'train mode at iteration {i}: {train_mode}')

            if mem_model.n_tasks > 1:
                mem_loss = trainEncoder(mem_model, inputs[ii:(ii+1)], learning_rate = learning_rate_mem, margin = margin, replay_prob = rpe, use_compressed = use_compressed)
                loss_task_id[mem_model.n_tasks-1].append(mem_loss)

            if train_mode == 'expert':

                # train the expert
                train_loss[i], train_acc[i] = trainScoreModule(score_model, inputs[ii:(ii+1)], labels[ii], optimizer_score, criterion, expert_decoder, criterion_autoencoder, lambda_autoencoder)

                # compute mean of the past `acc_window` trials
                # recent_acc[:-1] = recent_acc[1:].clone()
                # recent_acc[-1] = train_acc[(i+1-acc_window):(i+1)].mean()
                # mean_acc = recent_acc[-1]
                mean_acc = train_acc[(i+1-acc_window):(i+1)].mean()

                # if the current average accuracy has decreased from
                # best accuracy by `new_task_threshold`, and the best_accuracy is
                # high enough (> `check_new_task_acc`), then predict a task-change
                if (i+1 in task_start_inds) or (((best_acc - mean_acc > new_task_threshold) and (best_acc > check_new_task_acc)) and not task_start_inds):
                    # we think a new task has been encountered
                    print(f'add task at trial {i}')
                    print(f'(mean accuracy = {mean_acc})')
                    print()
                    time.sleep(1.)
                    addTask(score_model, mem_model, optimizer_score, criterion, replay_after_new)

                    params = [
                                {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                                'lr':learning_rate_ensembler,
                                'weight_decay':wd},
                                {'params':[param for expert in score_model.experts for param in expert.parameters()],
                                'lr':learning_rate_experts,
                                'weight_decay':wd}
                            ]
                    optimizer_score = optimizer(params, amsgrad=amsgrad)
                    optimizer_score.zero_grad()

                    train_mode = 'context_recognizer'

                    # initialize RL reward for each task
                    Q_task, Q_context, Q_overall =\
                        initializeEnsemblerRewards(score_model,
                                                   mem_model,
                                                   inputs[ii:(ii+1)],
                                                   use_compressed,
                                                   k,
                                                   learning_rate_context_td,
                                                   context_reward_weight)

                    # update tracked variables
                    within_mode_idx = 0
                    within_task_idx = 0
                    # best_acc = chance_acc
                    # recent_acc[:] = chance_acc
                    task_number += 1
                    loss_task_id[mem_model.n_tasks-1] = []
                else:
                    within_mode_idx += 1
                    within_task_idx += 1

            elif train_mode == 'context_recognizer':
                # print(Q_overall)
                # train the context recognizer
                train_loss[i], train_acc[i], Q_task[:], Q_context[:], Q_overall[:] = \
                    trainContextRecognizer(score_model,
                                           mem_model,
                                           inputs[ii:(ii+1)],
                                           labels[ii],
                                           optimizer_score,
                                           criterion,
                                           learning_rate_td,
                                           learning_rate_context_td,
                                           context_reward_weight,
                                           use_compressed,
                                           Q_task,
                                           Q_context,
                                           Q_overall)

                # recent_acc[:-1] = recent_acc[1:].clone()
                # recent_acc[-1] = train_acc[(i+1-acc_window):(i+1)].mean()
                # mean_acc = recent_acc[-1]
                mean_acc = train_acc[(i+1-acc_window):(i+1)].mean()

                if within_mode_idx == (n_try_old - 1):
                    # print('reinstate task set ' + str(int(best_action)))

                    # select the best Ensembler for the remainder of the new task context or create new one
                    action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
                    best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))].item()

                    # worst accuracy for current task is the average accuracy during the context recognizer phase
                    worst_acc = chance_acc
                    # worst_acc = train_acc[(i+1-n_try_old):(i+1)].mean()

                    # if worst_acc < perf_check_context_recognizer:
                    #     print(f'add ensembler at trial {i}')
                    #     print(f'(mean context recognizer accuracy = {worst_acc})')
                    #     # print(recent_acc)
                    #     print()
                    #     time.sleep(1.)
                    #     score_model.add_ensembler()
                    #     if worst_acc >= reuse_ensembler_threshold:
                    #         score_model.ensemblers[-1].load_state_dict(copy.deepcopy(score_model.ensemblers[best_action].state_dict()))
                    #     best_action = score_model.num_ensemblers - 1
                    # else:
                    #     print(f'Reuse ensembler {best_action}')
                    #     print()
                    #     time.sleep(1.)
                    #     # score_model.unfreeze_ensembler([best_action])

                    # score_model.task2ensembler.append(best_action)

                    print(f'Best ensembler: {best_action}')
                    score_model.ensemblers[-1].load_state_dict(copy.deepcopy(score_model.ensemblers[best_action].state_dict()))
                    score_model.task2ensembler.append(score_model.num_ensemblers - 1)

                    # switch to ensemble mode
                    train_mode = 'ensembler'
                    # print('train ensembler')
                    # time.sleep(1.)
                    within_mode_idx = 0

                    # increase ensemble learning rate and freeze experts
                    params = [
                        {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                         'lr':learning_rate_ensembler,
                         'weight_decay':wd},
                        {'params':[param for expert in score_model.experts for param in expert.parameters()],
                         'lr':learning_rate_experts,
                         'weight_decay':wd}
                    ]
                    optimizer_score = optimizer(params, amsgrad=amsgrad)
                    optimizer_score.zero_grad()

                    score_model.freeze_expert(list(range(score_model.num_experts)))

                else:
                    within_mode_idx += 1

                within_task_idx += 1

            # try tuning the ensembler network
            elif train_mode == 'ensembler':

                train_loss[i], train_acc[i] = trainScoreModule(score_model,
                                                             inputs[ii:(ii+1)],
                                                             labels[ii],
                                                             optimizer_score,
                                                             criterion)

                if replay_old_task:
                    if torch.rand((1,)) < expert_replay_prob:
                        replayTask(score_model, mem_model, optimizer_score, criterion)

                # compute mean of the past `acc_window` trials
                # recent_acc[:-1] = recent_acc[1:].clone()
                # recent_acc[-1] = train_acc[(i+1-acc_window):(i+1)].mean()
                # mean_acc = recent_acc[-1]
                mean_acc = train_acc[(i+1-acc_window):(i+1)].mean()

                if within_mode_idx == (epochs_ensembler_check-1):
                    # ensembler_acc = train_acc[(i+1-epochs_ensembler_check):(i+1)].mean()
                    # acc_slope = recent_acc[-epochs_ensembler_check:].diff().mean()
                    # if (ensembler_acc < (worst_acc + perf_check_ensembler)) and (acc_slope < perf_slope_check):
                    improvement_room = 1 - worst_acc
                    if (mean_acc < (worst_acc + perf_check_ensembler * improvement_room)) and (score_model.num_experts < max_experts):
                    # if ensembler_acc < (worst_acc + perf_check_ensembler):
                        # print('ensemble performance is insufficient: recruit new expert at iteration {}'.format(str(i)))
                        # time.sleep(1.)
                        train_mode = 'expert'
                        within_mode_idx = 0
                        print(f'add expert at trial {i}')
                        # print(f'(mean ensembler accuracy = {ensembler_acc})')
                        print(f'(mean accuracy = {mean_acc})')
                        # print(f'(avg increase in acc per trial = {acc_slope})')
                        # print(f'(recent avg accuracies = {recent_acc})')
                        # print(f'worst acc = {worst_acc}')
                        # print(f'acc slope = {acc_slope}')
                        print()
                        score_model.add_expert()
                        if expert_decoder:
                            params = [
                                {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                                'lr':learning_rate_experts,
                                'weight_decay':wd},
                                {'params':[param for expert in score_model.experts for param in expert.parameters()],
                                'lr':learning_rate_experts,
                                'weight_decay':wd},
                                {'params':[param for param in expert_decoder.parameters()],
                                'lr':learning_rate_experts,
                                'weight_decay':wd}
                            ]
                        else:
                            params = [
                                {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                                'lr':learning_rate_experts,
                                'weight_decay':wd},
                                {'params':[param for expert in score_model.experts for param in expert.parameters()],
                                'lr':learning_rate_experts,
                                'weight_decay':wd}
                            ]
                        optimizer_score = optimizer(params, amsgrad=amsgrad)
                        optimizer_score.zero_grad()
                    else:
                        within_mode_idx += 1
                        replay_old_task = True
                        score_model.unfreeze_expert([i for i in range(score_model.num_experts)])
                        params = [
                            {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                            'weight_decay':wd,
                            'lr':learning_rate_ensembler
                            },
                            {'params':[param for expert in score_model.experts for param in expert.parameters()],
                            'weight_decay':wd,
                            'lr':learning_rate_unfreeze
                            }
                        ]
                        optimizer_score = optimizer(params, amsgrad=amsgrad)
                        optimizer_score.zero_grad()
                        print(f'unfreeze experts at trial {i}')
                        # print(f'(mean ensembler accuracy = {ensembler_acc})')
                        print(f'(mean accuracy = {mean_acc})')
                        # print(f'(avg increase in acc per trial = {acc_slope})')
                        print()
                        time.sleep(1.)
                    within_task_idx += 1
                elif within_mode_idx > (epochs_ensembler_check - 1):
                    if (i+1 in task_start_inds) or (((best_acc - mean_acc > new_task_threshold) and (best_acc > check_new_task_acc)) and not task_start_inds):
                        # we think a new task has been encountered
                        print(f'add task at trial {i}')
                        print(f'(mean accuracy = {mean_acc})')
                        print()                        
                        addTask(score_model, mem_model, optimizer_score, criterion, replay_after_new)

                        params = [
                                {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                                'lr':learning_rate_experts,
                                'weight_decay':wd},
                                {'params':[param for expert in score_model.experts for param in expert.parameters()],
                                'lr':learning_rate_experts,
                                'weight_decay':wd}
                            ]
                        optimizer_score = optimizer(params, amsgrad=amsgrad)
                        optimizer_score.zero_grad()

                        train_mode = 'context_recognizer'

                        # initialize RL reward for each task
                        Q_task, Q_context, Q_overall = \
                            initializeEnsemblerRewards(score_model,
                                                       mem_model,
                                                       inputs[ii:(ii+1)],
                                                       use_compressed,
                                                       k,
                                                       learning_rate_context_td,
                                                       context_reward_weight)

                        # update tracked variables
                        within_mode_idx = 0
                        within_task_idx = 0
                        best_acc = chance_acc
                        # recent_acc[:] = chance_acc
                        task_number += 1
                        loss_task_id[mem_model.n_tasks-1] = []
                    else:
                        within_mode_idx += 1
                        within_task_idx += 1
                else:
                  within_mode_idx += 1
                  within_task_idx += 1

            # update the running best accuracy if it has increased
            if mean_acc > best_acc:
                best_acc = mean_acc

            # store memories
            mem_model.add_memory(inputs[ii], labels[ii])

            # perf = mean_acc

            # if i < 384:
            #     print(f'Mean acc at trial {i}: {perf}')
            #     print(score_model.task2ensembler)

            # if plot_perf:
            #     # plt.axis([0, n_train+1, 0, 1.])
            #     # # plt.xticks(list(range(num_epochs+1)))
            #     # plt.xticks([])
            #     # plt.xlabel('Training Example')
            #     # plt.ylabel('Accuracy')
            #     # plt.gcf().set_size_inches(15, 5)
            #     plt.scatter(i, perf, color=plot_colors[task_number], marker=plot_markers[score_model.num_experts-1])
            #     # plt.title(f'Trial {trial}\nIteration {i}/{n_train}', fontsize=20)
            #     display.display(plt.gcf())
            #     display.clear_output(wait=True)

            # if not i % print_every:
            #     print('SCORE {}/{} iterations complete'.format(i, n_train))

            i += 1

    print(f'(mean accuracy = {mean_acc})')

    return train_loss, train_acc, loss_task_id

def trainScoreModule(score_model, inputs, labels, optimizer, criterion, expert_decoder=None, criterion_autoencoder=None, lambda_autoencoder=0., task_idx=-1):

    preds, expert_preds = score_model(inputs)
    ensembler_preds = getEnsemblerPreds(score_model, score_model.task2ensembler[task_idx], preds)

    classification_loss = criterion(ensembler_preds.view(1, -1), labels.view(-1))
    train_loss = classification_loss.item()
    train_acc = (ensembler_preds.view(-1).topk(1)[1] == labels.view(-1)).float()

    if expert_decoder and lambda_autoencoder > 0.:
        decoder_input = expert_preds[:, :, score_model.expert_start_indices[-2]:score_model.expert_start_indices[-1]].view(inputs.size(0), -1)
        decoder_output = expert_decoder(decoder_input)
        reconstruction_error = criterion_autoencoder(decoder_output, inputs).view(inputs.size(0), -1).mean(dim=1).sum()
        reconstruction_error /= inputs.size(0)
        loss = classification_loss + lambda_autoencoder * reconstruction_error
        # print(classification_loss)
        # print(reconstruction_error)
        # print(lambda_autoencoder)
        # print()
    else:
        loss = classification_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_loss, train_acc

def addTask(score_model, mem_model, optimizer, criterion, replay_after_new):
    # replay previous task memories before instantiating new task set
    if replay_after_new:
        for mem_idx in range(mem_model.max_memories*(mem_model.n_tasks-1), mem_model.max_memories*mem_model.n_tasks):
            replay_mem = mem_model.memories['input'][mem_idx].view(1, -1)
            pred_mem = score_model(replay_mem)
            pred_mem = getEnsemblerPreds(score_model, score_model.task2ensembler[-1], pred_mem)
            loss = criterion(pred_mem.view(1, -1), mem_model.memories['target'][mem_idx].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    score_model.add_ensembler()

    score_model.freeze_expert(list(range(score_model.num_experts)))
    score_model.freeze_ensembler(list(range(score_model.num_ensemblers-1)))

    mem_model.add_task()

    # # if we don't already have an unused randomly initialized ensembler, then add a new one
    # if (score_model.num_ensemblers-1) in score_model.task2ensembler:
    #     score_model.add_ensembler()

    # print('new task at iteration {}'.format(str(i)))
    # print('train reinforcement')
    # time.sleep(1.)

    # within_mode_idx = 0
    # within_task_idx = 0
    # best_acc = chance_acc

    # task_number += 1

    # loss_task_id[mem_model.n_tasks-1] = []

def initializeEnsemblerRewards(score_model, mem_model, inputs, use_compressed, k, learning_rate_context_td, context_reward_weight):
    # initialize expected rewards for each task set and each context
    Q_task = torch.zeros(score_model.num_ensemblers)
    Q_context = torch.zeros(score_model.num_ensemblers)
    Q_overall = torch.zeros(score_model.num_ensemblers)

    # initialize ensembler parameters with nearest neighbor old task
    label_prob = predictNEM(mem_model, inputs, use_compressed, k).squeeze()
    ensembler_prob = torch.zeros_like(Q_context)
    for task in range(mem_model.n_tasks-1):
        ensembler_idx = score_model.task2ensembler[task]
        ensembler_prob[ensembler_idx] += label_prob[task]
    for ensembler_idx in range(score_model.num_ensemblers-1):
        Q_context[ensembler_idx] = (1 - learning_rate_context_td)*Q_context[ensembler_idx] + learning_rate_context_td*ensembler_prob[ensembler_idx] # initialize tast set reward with task probability based on context

    # initialize reward of new tentative task set
    # Q_context[-1] = torch.max(Q_context[:-1])

    Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

    return Q_task, Q_context, Q_overall

def trainContextRecognizer(score_model, mem_model, inputs, labels, optimizer, criterion, learning_rate_td, learning_rate_context_td, context_reward_weight, use_compressed, Q_task, Q_context, Q_overall):
    nearest_task = int(predictNEM(mem_model, inputs, use_compressed, 1).topk(1)[1].squeeze().numpy())

    q_update = torch.zeros_like(Q_context)
    if nearest_task < (mem_model.n_tasks - 1):
        ensembler_idx = score_model.task2ensembler[nearest_task]
        q_update[ensembler_idx] = 1
    Q_context[:] = (1 - learning_rate_context_td)*Q_context + learning_rate_context_td*q_update

    Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

    # select action that will give the highest expected reward
    action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
    best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]

    # get predictions from all ensemblers
    # score_model.ensemblers.eval()
    # with torch.no_grad():
    preds, expert_preds = score_model(inputs)
    score_model.ensemblers.train()

    # compare predictions to target label and update the rewards for each ensembler
    # print(f'Q_task before: {Q_task}')
    all_accs= torch.zeros(score_model.num_ensemblers, dtype=torch.long)
    for ensembler_idx in range(score_model.num_ensemblers):
        ensembler_pred = getEnsemblerPreds(score_model, ensembler_idx, preds)
        current_loss = criterion(ensembler_pred.view(1, -1), labels.view(-1))
        if ensembler_idx == score_model.num_ensemblers - 1:
            current_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        current_acc = (ensembler_pred.view(-1).topk(1)[1] == labels.view(-1)).float().mean()
        all_accs[ensembler_idx] = current_acc
        Q_task[ensembler_idx] = (1 - learning_rate_td)*Q_task[ensembler_idx] + learning_rate_td*current_acc.item()
        # store the loss for the ensembler that was actually employed
        if ensembler_idx == best_action:
            train_loss = current_loss.clone().item()
            train_acc = current_acc.clone()

    # print(all_accs)
    # print(f'Q_task after: {Q_task}')
    # print()
    # update Q_overall with new Q_task
    Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

    return train_loss, train_acc, Q_task, Q_context, Q_overall


def getEnsemblerPreds(score_model, ensembler_idx, preds):
    start_idx = score_model.ensembler_start_indices[ensembler_idx]
    end_idx = score_model.ensembler_start_indices[ensembler_idx+1]
    return preds[:, start_idx:end_idx]


def replayTask(score_model, mem_model, optimizer, criterion):
    replay_task_ind = torch.randint(0, mem_model.n_tasks-1, size=(1,)).item()
    replay_mem_ind = torch.randint(low=replay_task_ind*mem_model.max_memories, high=(replay_task_ind+1)*mem_model.max_memories, size=(1,)).item()
    mem_input = mem_model.memories['input'][replay_mem_ind:(replay_mem_ind+1)]
    mem_target = mem_model.memories['target'][replay_mem_ind]
    _, _ = trainScoreModule(score_model, mem_input, mem_target, optimizer, criterion, task_idx=replay_task_ind)


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

def trainSimple(model, feature_extractor, dataloader, hyper, cre=False):

    learning_rate_experts = hyper['learning_rate_experts']
    learning_rate_ensembler = hyper['learning_rate_ensembler']
    amsgrad = hyper['amsgrad']

    n_train = len(dataloader.dataset)

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in model.heads.parameters()], 'lr':learning_rate_ensembler},
        {'params':[param for param in model.experts.parameters()], 'lr':learning_rate_experts}
    ]
    optimizer = optim.Adam(params, amsgrad=amsgrad)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    i = 0

    for inputs, _, labels, task_ids in dataloader:
        inputs = inputs.to(device)
        if feature_extractor:
            with torch.no_grad():
                inputs = feature_extractor(inputs)
        labels = labels.to(device)

        batch_size = len(labels)

        for ii in range(batch_size):    
            pred = model(inputs[ii:(ii+1)], task_ids[ii])
            loss = criterion(pred.view(1, -1), labels[ii].view(-1))
            train_loss[i] = loss.item()
            train_acc[i] = (pred.view(-1).topk(1)[1] == labels[ii].view(-1)).float()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if not i % print_every:
            #     print('FineTune {}/{} iterations complete'.format(i, n_train))

            i += 1

    return train_loss, train_acc

def trainEWC(model, feature_extractor, dataloader, hyper, exclude_tasks=[]):
    learning_rate_experts = hyper['learning_rate_experts']
    learning_rate_ensembler = hyper['learning_rate_ensembler']
    lambda_ewc = hyper['lambda_ewc']
    amsgrad = hyper['amsgrad']

    n_train = len(dataloader.dataset)

    print_every = math.floor(n_train/10)

    # set training parameters
    params = [
        {'params':[param for param in model.heads.parameters()], 'lr':learning_rate_ensembler},
        {'params':[param for param in model.experts.parameters()], 'lr':learning_rate_experts}
    ]
    optimizer_net = optim.Adam(params, amsgrad=amsgrad)
    criterion = torch.nn.NLLLoss()
    optimizer_net.zero_grad()

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    i = 0

    for inputs, _, labels, task_ids in dataloader:
        inputs = inputs.to(device)
        if feature_extractor:
            with torch.no_grad():
                inputs = feature_extractor(inputs)
        labels = labels.to(device)

        batch_size = len(labels)

        for ii in range(batch_size):
            pred = model(inputs[ii:(ii+1)], task_ids[ii])
            loss_nll = criterion(pred.view(1, -1), labels[ii].view(-1))
            loss_ewc = regEWC(model, exclude_tasks)
            loss = loss_nll + lambda_ewc*loss_ewc
            train_loss[i] = loss.item()
            train_acc[i] = (pred.view(-1).topk(1)[1] == labels[ii].view(-1)).float()

            loss.backward()
            optimizer_net.step()
            optimizer_net.zero_grad()


            # if not i % print_every:
            #     print('EWC {}/{} iterations complete'.format(i, n_train))

            i += 1

    return train_loss, train_acc

def regEWC(model, exclude_tasks):
#     cur_params = [param for param in model.parameters()]
    loss_ewc = 0
    for task_id, _ in enumerate(model.task_params.keys()):
        if task_id not in exclude_tasks:
            for layer_ind, param in enumerate(model.experts.parameters()):
                loss_ewc += 0.5*(model.F[task_id][layer_ind]*(param - model.task_params[task_id][layer_ind])**2).squeeze().sum()
    return loss_ewc