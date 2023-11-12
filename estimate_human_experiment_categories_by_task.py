from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop
from torchvision.transforms.functional import crop
from IPython import display
from sklearn.decomposition import PCA
from PIL import Image
from io import open
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
import os
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# define training and validation data sets
img_size = 256
mytransform = Compose([
    Resize(img_size),
    ToTensor()
])

def getImageTensors(img_dir, regex):
    r = re.compile(regex)
    img_names = list(filter(r.match, os.listdir(img_dir)))
    if 'task_3' in regex:
        img_dir += 'task_3_'
    elif 'task_4' in regex:
        img_dir += 'task_4_'
    num_images = len(img_names)
    img_paths = [img_dir + f'stim_{i}.png' for i in range(num_images)]
    image_tensor = []
    for pth in img_paths:
        image = Image.open(pth).convert('L')
        image_tensor.append(mytransform(image).view(1, 1, img_size, img_size))
    return torch.cat(image_tensor, dim=0)


# initialize gabor feature extractor
frequency_extractor = GaborFeatureExtractor(img_size).to(device)
frequency_extractor.load_state_dict(torch.load('models/gabor-frequency-extractor-human-experiment.pt', map_location=device))

orientation_extractor = GaborFeatureExtractor(img_size).to(device)
orientation_extractor.load_state_dict(torch.load('models/gabor-orientation-weighting-extractor-human-experiment.pt', map_location=device))

feature_extractor = FeatureExtractor(frequency_extractor, orientation_extractor)
feature_extractor.eval()


def predsByRule(gabor_features):
    preds_by_rule = torch.zeros((inputs.size(0), 4))
    # Frequency classification rule
    pos_idx = gabor_features[:, 0] > 0.5
    preds_by_rule[pos_idx, 0] = 1

    # Orientation classification rule
    pos_idx = gabor_features[:, 1] > 0.5
    preds_by_rule[pos_idx, 1] = 1

    # Conjunction classification rule
    pos_idx = (gabor_features[:, 0] < 0.5) & (gabor_features[:, 1] < 0.5)
    preds_by_rule[pos_idx, 2] = 1

    # Information Integration classification rule
    pos_idx = (gabor_features[:, 0] + gabor_features[:, 1]) > 1.0
    preds_by_rule[pos_idx, 3] = 1

    return preds_by_rule

# get predictions made by each rule for the Task 1+2 stimuli
inputs = getImageTensors('/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/', "stim_\d*.png")
targets = torch.from_numpy(np.genfromtxt('/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/category_labels.csv', delimiter=",", dtype=np.int64))
with torch.no_grad():
    gabor_features = torch.sigmoid(feature_extractor(inputs.to(device)))

preds_by_rule = predsByRule(gabor_features)

pred_match = (preds_by_rule[:, 0] == targets[:, 0]).float().mean().item()
print(f'Task 1 match: {pred_match:0.3f}')

pred_match = (preds_by_rule[:, 1] == targets[:, 1]).float().mean().item()
print(f'Task 2 match: {pred_match:0.3f}')

# write preds_by_rule to file
np.savetxt(f'/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/task_1+2_predictions_by_rule.csv',
           preds_by_rule,
           delimiter=',',
           fmt ='%.0f',
           header='Frequency,Orientation,Conjunction,Information-Integration')


# get predictions made by each rule for the Task 3 stimuli
inputs = getImageTensors('/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/', "task_3_stim_\d*.png")
targets = torch.from_numpy(np.genfromtxt('/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/task_3_category_labels.csv', delimiter=",", dtype=np.int64))
with torch.no_grad():
    gabor_features = torch.sigmoid(feature_extractor(inputs.to(device)))

preds_by_rule = predsByRule(gabor_features)

pred_match = (preds_by_rule[:, 2] == targets[:]).float().mean().item()
print(f'Task 3 match: {pred_match:0.3f}')

# write preds_by_rule to file
np.savetxt(f'/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/task_3_predictions_by_rule.csv',
           predsByRule(gabor_features),
           delimiter=',',
           fmt ='%.0f',
           header='Frequency,Orientation,Conjunction,Information-Integration')

# get predictions made by each rule for the Task 4 stimuli
inputs = getImageTensors('/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/', "task_4_stim_\d*.png")
targets = torch.from_numpy(np.genfromtxt('/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/task_4_category_labels.csv', delimiter=",", dtype=np.int64))
with torch.no_grad():
    gabor_features = torch.sigmoid(feature_extractor(inputs.to(device)))

preds_by_rule = predsByRule(gabor_features)

pred_match = (preds_by_rule[:, 3] == targets[:]).float().mean().item()
print(f'Task 3 match: {pred_match:0.3f}')

# write preds_by_rule to file
np.savetxt(f'/Users/tyler/online-experiments/Human-Continual-Learning-Categorization/static/stims/task_4_predictions_by_rule.csv',
           predsByRule(gabor_features),
           delimiter=',',
           fmt ='%.0f',
           header='Frequency,Orientation,Conjunction,Information-Integration')