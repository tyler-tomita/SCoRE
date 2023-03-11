import torch
import torch.nn as nn
import torch.nn.functional as F

class GaborFeatureExtractor(nn.Module):
    def __init__(self, img_size, batchnorm=True):
        super(GaborFeatureExtractor, self).__init__()

        self.fc_in_size = int(256 * img_size/16 * img_size/16)
        if batchnorm:
            self.base = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                # output size = (32, 32, 16)
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                # output size = (16, 16, 32)
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                # output size = (8, 8, 64)
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                # output size = (4, 4, 128)
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                # output size = (2, 2, 256)
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Flatten()
            )
            self.fc = nn.Sequential(
                nn.Linear(self.fc_in_size, 100),
                nn.ReLU(),
                nn.BatchNorm1d(100),
                nn.Linear(100, 10),
                nn.ReLU(),
                nn.BatchNorm1d(10)
            )
            self.head = nn.Linear(10, 1)
        else:
            self.base = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                # output size = (32, 32, 16)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                # output size = (16, 16, 32)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                # output size = (8, 8, 64)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                # output size = (4, 4, 128)
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                # output size = (2, 2, 256)
                nn.ReLU(),
                nn.Flatten()
            )
            self.fc = nn.Sequential(
                nn.Linear(self.fc_in_size, 100),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(),
            )
            self.head = nn.Linear(10, 1)
        
    def forward(self, x):
        outputs = self.base(x)
        outputs = self.fc(outputs)
        outputs = self.head(outputs)
        return outputs