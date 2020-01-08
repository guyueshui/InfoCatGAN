import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from models.official_mnist import FrontD, Q
from utils import get_data

params = torch.load('results/MNIST/0.22%super/checkpoint/model-final.pt')
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
  dataset = dsets.MNIST('../datasets', train=False, transform=transforms.ToTensor())
  print(len(dataset))
  loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)
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
