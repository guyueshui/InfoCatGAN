#!/opt/py36/bin/python

import argparse
import torch
from torchvision import datasets
from torchvision import transforms

from trainer import *

def main(config):
  np.set_printoptions(precision=4)

  if config.dataset == 'mnist':
    import models.mnist as nets
    trans = transforms.ToTensor()
    dataset = datasets.MNIST(config.data_root, transform=trans, download=True)
  elif config.dataset == 'stl10':
    import models.stl10 as nets
    trans = transforms.ToTensor()
    dataset = datasets.STL10(config.data_root, transform=trans, download=True)
  else:
    raise NotImplementedError

  if config.imbalanced:
    sample_probs = np.random.rand(config.num_class)
    ib = ImbalanceSampler(dataset, sample_probs)
    dataset, true_dist = ib.ImbalancedDataset()
    print("The imbalanced dist is: ", true_dist)
  else:
    true_dist = config.cat_prob

  fd = nets.FrontD()
  d = nets.D()
  q = nets.Q()
  fg = nets.FrontG()
  g = nets.Generator()
  gt = nets.GTransProb()

  for i in [fd, d, q, fg, g, gt]:
    i.to(config.device)
    i.apply(weights_init)

  t = Trainer(config, dataset, fg, g, gt, fd, d, q)
  t.train(config.cat_prob, true_dist)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epoch', type=int, default=100)
  parser.add_argument('--num_class', type=int, default=10, help='Number of different categories.')
  parser.add_argument('--num_noise_dim', type=int, default=62, help='Uncompressed noise dimension (i.e., dim of z) of the generator input.')
  parser.add_argument('--batch_size', type=int, default=100)
  parser.add_argument('--tiny', type=float, default=1e-6, help='Global precison.')
  parser.add_argument('--dataset', type=str, default='mnist', help='The dataset to train.')
  parser.add_argument('--data_root', type=str, default='../datasets')
  parser.add_argument('--gpu', type=bool, default=True, help='Whether to use GPU.')
  parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  parser.add_argument('--use_ba', type=bool, default=False, help='Wether use BlahutArimoto algo.')
  parser.add_argument('--imbalanced', type=bool, default=False, help='Wether use processed to be imbalanced.')
  parser.add_argument('--cat_prob', default=np.array([0.1]).repeat(10), help='Default categorical distribution.')
  parser.add_argument('--experiment_tag', type=str, default='unifc', help='Experiment tag that gives some meaningful info.')
  config = parser.parse_args()
  print(config)

  # Fix random seeds.
  np.random.seed(2333)
  torch.manual_seed(2333)
  main(config)