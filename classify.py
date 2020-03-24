import argparse
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from utils import get_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MNIST")
args = parser.parse_args()

params = torch.load('results/' + args.dataset + '/0.2super/checkpoint/model-epoch-100.pt')
if args.dataset == "MNIST":
  from models.official_mnist import FrontD, Q
  dataset = dsets.MNIST('../datasets', train=False, transform=transforms.ToTensor())
elif args.dataset == "CIFAR10":
  from models.cifar10 import FrontD, Q
  dataset = dsets.CIFAR10('../datasets', train=False, transform=transforms.ToTensor())
else:
  raise NotImplementedError

fd = FrontD()
q = Q()
fd.load_state_dict(params['FrontD'])
q.load_state_dict(params['Q'])

def Classify(imgs):
  with torch.no_grad():
    logits, _, _ = q(fd(imgs))
    logits = logits.numpy()
  predicted = np.argmax(logits, axis=1)
  return predicted

if __name__ == '__main__':
  # Use test dataset.
  print(len(dataset))
  loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1)
  abatch = next(iter(loader))
  images, labels = abatch

  num_correct = 0
  for num_iter, (images, labels) in enumerate(loader):
    predicted = Classify(images)
    with torch.no_grad():
      labels = labels.numpy()

    loss = abs(predicted - labels)
    _, cnt = np.unique(loss, return_counts=True)
    batch_num_correct = cnt[0]
    print('Matched: ', batch_num_correct)
    num_correct += batch_num_correct
    
  acc = num_correct / len(dataset)
  print('Accuracy is: ', acc)