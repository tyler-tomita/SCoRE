import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from generate_dataset import *

class GaborDataset(Dataset):
    """Gabor Continual Learning dataset."""

    def __init__(self, root_dir, generate_data=False, train=True, transform=None):

        self.root_dir = root_dir

        if generate_data:
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            newpath = os.path.join(root_dir, 'train')
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            newpath = os.path.join(root_dir, 'train', 'images')
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            newpath = os.path.join(root_dir, 'val')
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            newpath = os.path.join(root_dir, 'val', 'images')
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            generate_dataset(root_dir)

        self.transform = transform
        if train:
            self.phase = 'train'
        else:
            self.phase = 'val'

        feature_path = os.path.join(self.root_dir,
                                    self.phase,
                                    'features.csv')
        self.stim_features = torch.from_numpy(np.genfromtxt(feature_path, delimiter=","))

        target_path = os.path.join(self.root_dir,
                                   self.phase,
                                   'targets.csv')
        self.targets = torch.from_numpy(np.genfromtxt(target_path, delimiter=","))

        task_path = os.path.join(self.root_dir,
                                    self.phase,
                                    'task_ids.csv')
        self.task_ids = torch.from_numpy(np.genfromtxt(task_path, delimiter=","))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.phase,
                                'images',
                                f'img_{idx}.png')
        img = Image.open(img_name).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        sample = [img, self.stim_features[idx], self.targets[idx], self.task_ids[idx]]

        return sample[0], sample[1], sample[2], sample[3]
    