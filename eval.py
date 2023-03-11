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

def eval_score(score_model, mem_model, input_tensor, target_tensor, task_label_tensor, num_tasks, use_compressed=False, k=1):
  n = input_tensor.shape[0]
  preds = torch.zeros((n, 2))
  task_label_preds = torch.zeros(n)
  unique_task_labels = torch.tensor(list(range(num_tasks)))

  for i in range(n):
    # initialize ensemble parameters with nearest neighbor old task
    task_label_preds[i] = predictNEM(mem_model, input_tensor[i], use_compressed, k).squeeze().topk(1)[1].squeeze()
    # predicted_task_set = task_label_preds[i].topk(1)[1].squeeze()
    for layer_idx, layer in enumerate([0, 2]):
        score_model.ensemble[layer].weight.data[:] = 0.
        score_model.ensemble[layer].bias.data[:] = 0.

    retrieved_weight = mem_model.memories['params'][int(task_label_preds[i])]['weight']
    retrieved_bias = mem_model.memories['params'][int(task_label_preds[i])]['bias']
    for layer_idx, layer in enumerate([0, 2]):
        retrieved_size = retrieved_weight[layer_idx].shape[1]
        score_model.ensemble[layer].weight.data[:, :retrieved_size] = retrieved_weight[layer_idx]
        score_model.ensemble[layer].bias.data[:] = retrieved_bias[layer_idx]

    with torch.no_grad():
      preds[i] = score_model(input_tensor[i, :, :])[1]

  preds = preds.to(device)
  task_label_preds = task_label_preds.to(device)
  acc = (preds.topk(1, dim=1)[1].squeeze() == target_tensor.squeeze()).float().mean()
  task_label_acc = (task_label_preds == task_label_tensor.squeeze()).float().mean()
  task_label_confusion = confusionMatrix(task_label_tensor.squeeze(), task_label_preds, num_tasks, normalize=False)

  return acc, task_label_acc, task_label_confusion
