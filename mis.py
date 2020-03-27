# This script is to compute inception score on MNIST dataset.
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(MNISTClassifier, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 32, 5), # 24
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(2),  # 12
      nn.Conv2d(32, 64, 5), # 8
      nn.MaxPool2d(2),  # 4
    )
    self.fc = nn.Sequential(
      nn.Linear(64*4*4, 1024),
      nn.ReLU(),
      nn.BatchNorm1d(1024),
      nn.Dropout(0.3),
      nn.Linear(1024, 10),
    )
  
  def forward(self, x):
    x = self.conv(x).view(-1, 64*4*4)
    x = self.fc(x)
    return x
