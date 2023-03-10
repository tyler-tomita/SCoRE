def trainSCORE(score_model, mem_model, data_loader, optimizer, score_hyper, mem_hyper, freeze_experts=True, use_compressed=False, plot_perf=True):

    learning_rate_experts = score_hyper['learning_rate_experts']
    learning_rate_ensembler = score_hyper['learning_rate_ensembler']
    learning_rate_unfreeze = score_hyper['learning_rate_unfreeze']
    wd = score_hyper['weight_decay']
    learning_rate_td = score_hyper['learning_rate_td']
    learning_rate_context_td = score_hyper['learning_rate_context_td']
    context_reward_weight = score_hyper['context_reward_weight']
    perf_slope_check = score_hyper['perf_slope_check']
    perf_check = score_hyper['perf_check']
    epochs_ensembler_check = score_hyper['epochs_ensembler_check']
    acc_window = score_hyper['acc_window']
    mix_ratio = score_hyper['mix_ratio'] # number in [0, 1] specifying how much to mix old memory ensemble with current ensemble
    n_try_old = score_hyper['n_try_old']
    new_task_threshold = score_hyper['new_task_threshold']
    check_new_task_acc = score_hyper['check_new_task_acc']
    replay_after_new = score_hyper['replay_after_new']

    learning_rate_mem = mem_hyper['learning_rate']
    margin = mem_hyper['margin']
    decoder_epochs = mem_hyper['epochs']
    replay_prob_encoder = mem_hyper['replay_prob_encoder']
    replay_prob_decoder = mem_hyper['replay_prob_decoder']
    k = mem_hyper['k']
    expert_replay_prob = mem_hyper['expert_replay_prob']

    rpe = replay_prob_encoder

    replay_old_task = False

    n_train = inputs.shape[0]

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
    params = [
        {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
         'lr':learning_rate_experts,
         'weight_decay':wd},
        {'params':[param for expert in score_model.experts for param in expert.parameters()],
         'lr':learning_rate_experts,
         'weight_decay':wd}
    ]
    optimizer_score = optimizer(params)
    optimizer_score.zero_grad()
    criterion = torch.nn.NLLLoss()

    loss_task_id = {}
    loss_task_id[mem_model.n_tasks-1] = []

    keep_training = True

    train_loss = torch.zeros(n_train)
    train_acc = torch.zeros(n_train)

    chance_acc = (labels == torch.mode(labels, dim=0)[0]).float().mean()
    recent_acc = (torch.rand(acc_window).to(device) < chance_acc).float()
    best_acc = chance_acc

    # if plot_perf:
    #         plt.axis([-1, n_train+1, 0, 1.05])
    #         plt.axvline(x=n_train/4, linestyle='--', color='gray')
    #         plt.axvline(x=n_train/4*2, linestyle='--', color='gray')
    #         plt.axvline(x=n_train/4*3, linestyle='--', color='gray')
    #         # plt.xticks(list(range(num_epochs+1)))
    #         plt.xticks([int(n_train/8)*ii for ii in range(9)], fontsize=20)
    #         plt.xlabel('Training Example', fontsize=20)
    #         plt.ylabel('Accuracy', fontsize=20)
    #         plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=20)
    #         plt.gcf().set_size_inches(15, 5)
    #         # plt.title(f'Trial {trial}\nIteration {0}/{n_train}', fontsize=20)
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)


    i = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        if feature_extractor:
            inputs = feature_extractor(inputs)
        labels = labels.to(device)

        batch_size = len(labels)

        for ii in range(batch_size):
            if mem_model.n_tasks > 1:
                mem_loss = trainEncoder(mem_model, inputs[ii], learning_rate = learning_rate_mem, margin = margin, replay_prob = rpe, use_compressed = use_compressed)
                loss_task_id[mem_model.n_tasks-1].append(mem_loss)

            if train_mode == 'expert':

                # train the expert
                train_loss[i], train_acc[i] = trainExpert(score_model, inputs[ii:(ii+1)], labels[ii], optimizer_score, criterion)

                # compute mean of the past `window_size` trials
                recent_acc[:-1] = recent_acc[1:].clone()
                recent_acc[-1] = train_acc[i]
                mean_acc = recent_acc.mean()

                # if the current average accuracy has decreased from
                # best accuracy by `new_task_threshold`, and the best_accuracy is
                # high enough (> `check_new_task_acc`), then predict a task-change
                if (best_acc - mean_acc > new_task_threshold) and (best_acc > check_new_task_acc):

                    # we think a new task has been encountered
                    addTask(score_model, mem_model, optimizer_score, criterion, replay_after_new)

                    train_mode = 'context_recognizer'

                    # initialize RL reward for each task
                    Q_task, Q_context, Q_overall = initializeEnsemblerRewards(
                        score_model, mem_model, inputs[ii], use_compressed, k,
                        learning_rate_context_td, context_reward_weight)

                    # update tracked variables
                    within_mode_idx = 0
                    within_task_idx = 0
                    best_acc = chance_acc
                    task_number += 1
                    loss_task_id[mem_model.n_tasks-1] = []
                else:
                    within_mode_idx += 1
                    within_task_idx += 1

            elif train_mode == 'context_recognizer':

                # train the context recognizer
                train_loss[i], train_acc[i], Q_task[:], Q_context[:], Q_overall[:] =
                trainContextRecognizer(score_model, mem_model, inputs, labels,
                                       use_compressed, Q_task, Q_context, Q_overall)

                if within_mode_idx == (n_try_old - 1):
                    # print('reinstate task set ' + str(int(best_action)))

                    # select the best Ensembler for the remainder of the new task context
                    action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
                    best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
                    score_model.task2ensembler.append(best_action)

                    # switch to ensemble mode
                    train_mode = 'ensembler'
                    # print('train ensemble')
                    # time.sleep(1.)
                    within_mode_idx = 0

                    # increase ensemble learning rate and freeze experts
                    params = [
                        {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                         'lr':learning_rate_ensembler,
                         'weight_decay':0.},
                        {'params':[param for expert in score_model.experts for param in expert.parameters()],
                         'lr':learning_rate_experts,
                         'weight_decay':wd}
                    ]
                    optimizer_score = optimizer(params)
                    optimizer_score.zero_grad()

                    score_model.freeze_expert(list(range(score_model.num_experts)))

                else:
                    within_mode_idx += 1

                recent_acc[:-1] = recent_acc[1:].clone()
                recent_acc[-1] = train_acc[i]
                mean_acc = recent_acc.mean()

                within_task_idx += 1

            # try tuning the ensembler network
            elif train_mode == 'ensembler':

                train_loss, train_acc = trainEnsembler(score_model, train_model,
                                                       inputs, labels,
                                                       criterion, score_optimizer)

                # TO DO
                # if replay_old_task:
                #     replayTask()

                # compute mean of the past `window_size` trials
                recent_acc[:-1] = recent_acc[1:].clone()
                recent_acc[-1] = train_acc[i]
                mean_acc = recent_acc.mean()

                if within_mode_idx == (epochs_ensemble_check-1):
                    perf_slope = torch.mean(train_acc[(i-acc_window):i]) - torch.mean(train_acc[:acc_window])
                    perf = torch.mean(train_acc[(i-acc_window):i])
                    # print('acc slope: ' + str(perf_slope))
                    # print('acc: ' + str(perf))
                    if perf < perf_check:
                        # print('ensemble performance is insufficient: recruit new expert at iteration {}'.format(str(i)))
                        # time.sleep(1.)
                        train_mode = 'expert'
                        within_mode_idx = 0
                        score_model.add_expert()
                        params = [
                            {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                             'lr':learning_rate_experts,
                             'weight_decay':wd},
                            {'params':[param for expert in score_model.experts for param in expert.parameters()],
                             'lr':learning_rate_experts,
                             'weight_decay':wd}
                        ]
                        optimizer_score = optimizer(params)
                        optimizer_score.zero_grad()
                    else:
                        within_mode_idx += 1
                        replay_old_task = True
                        score_model.unfreeze_expert([i for i in range(score_model.num_experts)])
                        params = [
                            {'params':[param for ensembler in score_model.ensemblers for param in ensembler.parameters()],
                            'weight_decay':0.,
                            'lr':learning_rate_ensembler
                            },
                            {'params':[param for expert in score_model.experts for param in expert.parameters()],
                            'weight_decay':wd,
                            'lr':learning_rate_unfreeze
                            }
                        ]
                        optimizer_score = optimizer(params)
                        optimizer_score.zero_grad()
                        # print('ensemble performance is sufficient: unfreeze experts')
                        # time.sleep(1.)
                    within_task_idx += 1
                elif within_mode_idx > (epochs_ensemble_check - 1):

                    if (best_acc - mean_acc > new_task_threshold) and (best_acc > check_new_task_acc):

                        # we think a new task has been encountered
                        addTask(score_model, mem_model, optimizer, criterion, replay_after_new)

                        train_mode = 'context_recognizer'

                        # initialize RL reward for each task
                        Q_overall = initializeEnsemblerRewards(score_model, mem_model, inputs[ii], use_compressed, k, learning_rate_context_td, context_reward_weight)

                        # update tracked variables
                        within_mode_idx = 0
                        within_task_idx = 0
                        best_acc = chance_acc
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

            idx_low = i-acc_window
            if idx_low < 0:
                idx_low = 0
            idx_high = i
            perf = torch.mean(train_acc[idx_low:idx_high])

            if plot_perf:
                # plt.axis([0, n_train+1, 0, 1.])
                # # plt.xticks(list(range(num_epochs+1)))
                # plt.xticks([])
                # plt.xlabel('Training Example')
                # plt.ylabel('Accuracy')
                # plt.gcf().set_size_inches(15, 5)
                plt.scatter(i, perf, color=plot_colors[task_number], marker=plot_markers[score_model.num_experts-1])
                # plt.title(f'Trial {trial}\nIteration {i}/{n_train}', fontsize=20)
                display.display(plt.gcf())
                display.clear_output(wait=True)
            # if not i % print_every:
            #     print('SCORE {}/{} iterations complete'.format(i, n_train))

        i += 1

    return train_loss, train_acc, loss_task_id

def trainExpert(score_model, inputs, labels, optimizer, criterion):
    # compute loss on current prediction
    pred = score_model(inputs)[1]
    current_loss = criterion(pred.view(1, -1), labels.view(-1))
    train_loss = current_loss.item()
    train_acc = (pred.view(-1).topk(1)[1] == labels.view(-1)).float().mean()

    # backpropagate and update parameters
    current_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_loss, train_acc


def addTask(score_model, mem_model, optimizer, criterion, replay_after_new):
    # replay previous task memories before instantiating new task set
    if replay_after_new:
        for mem_idx in range(mem_model.max_memories*(mem_model.n_tasks-1), mem_model.max_memories*mem_model.n_tasks):
            replay_mem = mem_model.memories['input'][mem_idx]
            pred_mem = score_model(replay_mem)[1]
            loss = criterion(pred_mem.view(1, -1), mem_model.memories['target'][mem_idx].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    score_model.freeze_expert(list(range(score_model.num_experts)))

    train_mode = 'reinforcement'

    mem_model.add_task()

    # if we don't already have an unused randomly initialized ensembler, then add a new one
    if (score_model.num_ensemblers-1) in score_model.task2ensembler:
        score_model.add_ensembler()

    # print('new task at iteration {}'.format(str(i)))
    # print('train reinforcement')
    # time.sleep(1.)

    # within_mode_idx = 0
    # within_task_idx = 0
    # best_acc = chance_acc

    # task_number += 1

    # loss_task_id[mem_model.n_tasks-1] = []

def initializeEnsemblerRewards(score_model, mem_model, inputs, use_compressed, k, learning_rate_context_td, context_reward_weight)
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

def trainContextRecognizer(score_model, mem_model, inputs, labels, use_compressed, Q_task, Q_context, Q_overall):
    nearest_task = int(predictNEM(mem_model, inputs.squeeze(), use_compressed, 1).topk(1)[1].squeeze().numpy())

    q_update = torch.zeros_like(Q_context)
    if nearest_task < (mem_model.n_tasks - 1):
        ensembler_idx = score_model.task2ensembler[nearest_task]
        q_update[ensembler_idx] = 1
    Q_context[ensembler_idx] = (1 - learning_rate_context_td)*Q_context[task] + learning_rate_context_td*q_update

    Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

    # select action that will give the highest expected reward
    action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
    best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]

    # get predictions from all ensemblers
    with torch.no_grad():
        preds = score_model(inputs)

    # compare predictions to target label and update the rewards for each ensembler
    for ensembler_idx in range(score_model.num_ensemblers):
        ensembler_pred = getEnsemblerPreds(score_model, ensembler_idx, preds)
        current_loss = criterion(ensembler_pred.view(1, -1), labels.view(-1))
        current_acc = (ensembler_pred.view(-1).topk(1)[1] == labels.view(-1)).float().mean()
        Q_task[ensembler_idx] = (1 - learning_rate_td)*Q_task[task] + learning_rate_td*current_acc[0]

        # store the loss for the ensembler that was actually employed
        if ensembler_idx == best_action:
            train_loss = current_loss
            train_acc = current_acc

    # update Q_overall with new Q_task
    Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context

    return train_loss, train_acc, Q_task, Q_context, Q_overall


def getEnsemblerPreds(score_model, ensembler_idx, preds):
    start_idx = score_model.ensemble_start_indices[ensembler_idx]
    end_idx = score_model.ensemble_start_indices[ensembler_idx+1]
    return preds[:, start_idx:end_idx]

def trainEnsembler(score_model, mem_model, inputs, labels, optimizer, criterion):

    preds = score_model(inputs)
    ensembler_preds = getEnsemblerPreds(score_model, score_model.task2ensembler[-1], preds)

    current_loss = criterion(ensembler_preds.view(1, -1), labels.view(-1))
    train_loss = current_loss.item()
    train_acc = (ensembler_preds.view(-1).topk(1)[1] == labels.view(-1)).float()

    current_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_loss, train_acc


# TO DO
# def replayTask(expert_replay_prob):
#     if torch.rand((1,)) < expert_replay_prob:
#         # select action that will give the highest expected reward
#         action_ind = torch.where(Q_overall == torch.max(Q_overall))[0]
#         best_action = action_ind[torch.randint(high=action_ind.shape[0], size=(1,))]
#         for task in range(mem_model.n_tasks-1):
#             retrieved_weight = mem_model.memories['params'][task]['weight']
#             retrieved_bias = mem_model.memories['params'][task]['bias']
#             for layer_idx, layer in enumerate([0, 2]):
#                 retrieved_size = retrieved_weight[layer_idx].shape[1]
#                 score_model.ensemble[layer].weight.data = torch.zeros_like(score_model.ensemble[layer].weight.data, device=device)
#                 score_model.ensemble[layer].weight.data[:, :retrieved_size] = retrieved_weight[layer_idx]
#                 score_model.ensemble[layer].bias.data = retrieved_bias[layer_idx]
#             with torch.no_grad():
#                 pred = score_model(inputs[ii:(ii+1)])[1]
#             current_acc = (pred.view(-1).topk(1)[1] == labels[ii].view(-1)).float()
#             Q_task[task] = (1 - learning_rate_td)*Q_task[task] + learning_rate_td*current_acc
#         Q_overall[:] = (1 - context_reward_weight)*Q_task + context_reward_weight*Q_context


#         task_replay_prob = Q_overall[:-1]/Q_overall[:-1].sum()
#         # replay_task_ind = int(torch.multinomial(task_replay_prob, 1))
#         replay_task_ind = int(task_replay_prob.topk(1)[1])

#         retrieved_weight = mem_model.memories['params'][replay_task_ind]['weight']
#         retrieved_bias = mem_model.memories['params'][replay_task_ind]['bias']
#         for layer_idx, layer in enumerate([0, 2]):
#             retrieved_size = retrieved_weight[layer_idx].shape[1]
#             score_model.ensemble[layer].weight.data = torch.zeros_like(score_model.ensemble[layer].weight.data, device=device)
#             score_model.ensemble[layer].weight.data[:, :retrieved_size] = retrieved_weight[layer_idx]
#             score_model.ensemble[layer].bias.data = retrieved_bias[layer_idx]
#         replay_mem_ind = torch.randint(low=replay_task_ind*mem_model.max_memories, high=(replay_task_ind+1)*mem_model.max_memories-1, size=(1,))
#         replay_mem = mem_model.memories['input'][int(replay_mem_ind)]
#         pred_mem = score_model(replay_mem)[1]
#         loss = criterion(pred_mem.view(1, -1), mem_model.memories['target'][int(replay_mem_ind)].view(-1))
#         loss.backward()
#         optimizer_score.step()
#         optimizer_score.zero_grad()

#         retrieved_weight = mem_model.memories['params'][mem_model.n_tasks-1]['weight']
#         retrieved_bias = mem_model.memories['params'][mem_model.n_tasks-1]['bias']
#         for layer_idx, layer in enumerate([0, 2]):
#             retrieved_size = retrieved_weight[layer_idx].shape[1]
#             score_model.ensemble[layer].weight.data = torch.zeros_like(score_model.ensemble[layer].weight.data, device=device)
#             score_model.ensemble[layer].weight.data[:, :retrieved_size] = retrieved_weight[layer_idx]
#             score_model.ensemble[layer].bias.data = retrieved_bias[layer_idx]

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
