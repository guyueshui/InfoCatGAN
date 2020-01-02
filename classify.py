import torch
from models.official_mnist import FrontD, Q

params = torch.load('results/MNIST/ss-first-try/checkpoint/model-final.pt')


if __name__ == '__main__':
  