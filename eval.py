import torch
from time import sleep
from train import predictNEM, getEnsemblerPreds
from utils import confusionMatrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def eval_score(score_model, mem_model, input_tensor, target_tensor, task_label_tensor, num_tasks, use_compressed=False, k=1):
#   n = input_tensor.shape[0]
#   preds = torch.zeros((n, 2))
#   task_label_preds = torch.zeros(n)
#   unique_task_labels = torch.tensor(list(range(num_tasks)))

#   for i in range(n):
#     # initialize ensemble parameters with nearest neighbor old task
#     task_label_preds[i] = predictNEM(mem_model, input_tensor[i], use_compressed, k).squeeze().topk(1)[1].squeeze()
#     # predicted_task_set = task_label_preds[i].topk(1)[1].squeeze()
#     for layer_idx, layer in enumerate([0, 2]):
#         score_model.ensemble[layer].weight.data[:] = 0.
#         score_model.ensemble[layer].bias.data[:] = 0.

#     retrieved_weight = mem_model.memories['params'][int(task_label_preds[i])]['weight']
#     retrieved_bias = mem_model.memories['params'][int(task_label_preds[i])]['bias']
#     for layer_idx, layer in enumerate([0, 2]):
#         retrieved_size = retrieved_weight[layer_idx].shape[1]
#         score_model.ensemble[layer].weight.data[:, :retrieved_size] = retrieved_weight[layer_idx]
#         score_model.ensemble[layer].bias.data[:] = retrieved_bias[layer_idx]

#     with torch.no_grad():
#       preds[i] = score_model(input_tensor[i, :, :])[1]

#   preds = preds.to(device)
#   task_label_preds = task_label_preds.to(device)
#   acc = (preds.topk(1, dim=1)[1].squeeze() == target_tensor.squeeze()).float().mean()
#   task_label_acc = (task_label_preds == task_label_tensor.squeeze()).float().mean()
#   task_label_confusion = confusionMatrix(task_label_tensor.squeeze(), task_label_preds, num_tasks, normalize=False)

#   return acc, task_label_acc, task_label_confusion

def eval_ewc(model, dataloader, head_idx, feature_extractor=None):
    # Get validation error
    # Iterate over data.
    model.eval()
    running_acc = 0.0
    n = 0
    for inputs, _, targets, _ in dataloader:
        batch_size = targets.size(0)
        n += batch_size
        inputs = inputs.to(device)
        if feature_extractor:
            with torch.no_grad():
                inputs = feature_extractor(inputs)

        # forward
        with torch.no_grad():
            preds = model(inputs, head_idx).view(batch_size, -1)
            acc = (preds.topk(1, dim=1)[1].squeeze() == targets.squeeze()).float().sum().item()

        # statistics
        running_acc += acc

    classification_acc = running_acc / n

    print(f'Classification Accuracy: {classification_acc:.4f}')
    print()

    return classification_acc

def eval_score(model, mem_model, dataloader, feature_extractor=None, infer_task=True):
    # Get validation error
    # Iterate over data.
    n = len(dataloader.dataset)
    model.eval()
    running_acc = 0.0
    running_task_label_acc = 0.0
    i = 0
    for inputs, _, targets, task_ids in dataloader:
        inputs = inputs.to(device)
        if feature_extractor:
            with torch.no_grad():
                inputs = feature_extractor(inputs)

        batch_size = len(targets)

        for ii in range(batch_size):
            with torch.no_grad():
              task_label_pred = predictNEM(mem_model, inputs[ii:(ii+1)], use_compressed=False, k=1).squeeze().topk(1)[1].squeeze()
              nearest_task = int(task_label_pred.numpy())
            task_label_acc = (task_label_pred == task_ids[ii]).float().item()
            
            if infer_task:
                ensembler_idx = model.task2ensembler[nearest_task]
            else:
                ensembler_idx = model.task2ensembler[task_ids[ii].item()]
            with torch.no_grad():
              pred, _ = model(inputs[ii])
            ensembler_pred = getEnsemblerPreds(model, ensembler_idx, pred.view(1, -1))
            acc = (ensembler_pred.view(-1).topk(1)[1] == targets[ii].view(-1)).float().item()
            i += 1

            # statistics
            running_acc += acc
            running_task_label_acc += task_label_acc

    classification_acc = running_acc / n
    task_inference_acc = running_task_label_acc / n

    print(f'Classification Accuracy: {classification_acc:.4f}')
    if infer_task:
        print(f'Task Inference Accuracy: {task_inference_acc:.4f}')
    print()

    return classification_acc, task_inference_acc


def eval_dynamoe(model, dataloader, feature_extractor=None):
    # Get validation error
    # Iterate over data.
    n = len(dataloader.dataset)
    model.eval()
    running_acc = 0.0
    i = 0
    for inputs, _, targets, task_ids in dataloader:
        inputs = inputs.to(device)
        if feature_extractor:
            with torch.no_grad():
                inputs = feature_extractor(inputs)

        batch_size = len(targets)

        for ii in range(batch_size):
            expert_idx = model.task2expert[task_ids[ii]] # the model is provided with a "task oracle" which tells it which task is being performed
            with torch.no_grad():
              all_preds = model(inputs[ii:(ii+1)])
            pred = all_preds[expert_idx]
            acc = (pred.view(-1).topk(1)[1] == targets[ii].view(-1)).float().item()
            i += 1

            # statistics
            running_acc += acc

    classification_acc = running_acc / n

    print(f'Classification Accuracy: {classification_acc:.4f}')
    print()

    return classification_acc