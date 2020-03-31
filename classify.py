import torch
import numpy as np

from utils import get_data
from config import get_config
from CatGAN import CatGAN

args = get_config()
path = 'results/' + args.dataset
path += '/re-ssinfogan'
path += '/model-epoch-70.pt'

dataset = get_data(args.dataset, args.data_root, train=False)
gan = CatGAN(args, dataset)
gan.load_model(path, gan.D)

if __name__ == '__main__':
  # Use test dataset.
  print(len(dataset))
  loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1)

  # Category matching step of CatGAN.
  abatch = next(iter(loader))
  images, labels = abatch
  images = images.to(gan.device)
  with torch.no_grad():
    yk = gan.D(images)
    yk = yk.numpy()
    labels = labels.numpy()
  assert len(yk) == len(labels)
  mat = np.zeros((len(yk), len(labels)), dtype=int)
  for i in range(len(yk)):
    mat[yk[i], labels[i]] += 1
  print(mat)
  map_to_real = np.argmax(mat, axis=1)
  print(map_to_real)

  num_correct = 0
  for num_iter, (images, labels) in enumerate(loader):
    images = images.to(gan.device)
    predicted = gan.classify(images, map_to_real)
    with torch.no_grad():
      labels = labels.numpy()

    assert predicted.shape == labels.shape
    loss = abs(predicted - labels)
    _, cnt = np.unique(loss, return_counts=True)
    batch_num_correct = cnt[0]
    print('Matched: ', batch_num_correct)
    num_correct += batch_num_correct
    
  acc = num_correct / len(dataset)
  print('Accuracy is: ', acc)
