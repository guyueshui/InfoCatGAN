#!/opt/py36/bin/python

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from trainer import Trainer
from utils import weights_init, get_data, ImbalanceSampler

def main(config):
  if config.dataset == 'MNIST':
    import models.official_mnist as nets

  elif config.dataset == 'FashionMNIST':
    import models.official_mnist as nets

  elif config.dataset == 'STL10':
    import models.stl10 as nets
    config.num_noise_dim == 256
    config.num_dis_c = 10

  elif config.dataset == 'CelebA':
    import models.celeba as nets
    config.num_noise_dim = 128
    config.num_dis_c = 10
    config.num_class = 10
    config.num_con_c = 0

  # elif config.dataset == 'SVHN':
  #   import models.svhn as nets
  #   config.num_noise_dim = 124
  #   config.num_dis_c = 4
  #   config.num_class = 10
  #   config.num_con_c = 4

  else:
    raise NotImplementedError

  dataset = get_data(config.dataset, config.data_root)

  if config.imbalanced:
    sample_probs = np.random.rand(config.num_class)
    idx = [i for i in range(10) if sample_probs[i] < 0.6]
    sample_probs[idx] = 0.6 # Normalize lead to fewer samples, that's not good.
    ib = ImbalanceSampler(dataset, sample_probs)
    dataset, true_dist = ib.ImbalancedDataset()
    print("The imbalanced dist is: ", true_dist)
  else:
    true_dist = np.array([1 / config.num_class]).repeat(config.num_class)

  if config.gpu == 0:  # GPU selection.
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  elif config.gpu == 1:
    config.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
  elif config.gpu == -1:
    config.device = torch.device('cpu')
  else:
    raise IndexError('Invalid GPU index')

  fd = nets.FrontD()
  d = nets.D()
  q = nets.Q()
  fg = nets.FrontG()
  g = nets.Generator()
  gt = nets.GTransProb()

  for i in [fd, d, q, fg, g, gt]:
    i.to(config.device)
    i.apply(weights_init)

  print(config)
  t = Trainer(config, dataset, fg, g, gt, fd, d, q)
  Glosses, Dlosses, EntQC_given_X, MSEs = t.train(config.cat_prob, true_dist)
  
  # Plotting losses...
  plt.figure(figsize=(10, 5))
  plt.title('GAN Loss')
  plt.plot(Glosses, label='G', linewidth=1)
  plt.plot(Dlosses, label='D', linewidth=1)
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(t._savepath + '/gan_loss.png')
  plt.close('all')

  plt.figure(figsize=(10, 5))
  plt.title('Entropy Loss')
  plt.plot(EntQC_given_X, linewidth=1)
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.savefig(t._savepath + '/ent_loss.png')
  plt.close('all')

  if config.use_ba:
    plt.figure(figsize=(10, 5))
    plt.title('RMSE')
    plt.plot(MSEs, linewidth=1)
    plt.savefig(t._savepath + '/rmse.png')
    plt.close('all')

if __name__ == '__main__':
  ##############################
  # Pre configs.
  ##############################
  from config import config
  np.set_printoptions(precision=4)

  # Fix random seeds.
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)

  # if config.use_ba or config.imbalanced:
  #   initc = np.array([0.147, 0.037, 0.033, 0.143, 0.136, 0.114, 0.057, 0.112, 0.143, 0.078])
  #   initc /= np.sum(initc)
  #   config.cat_prob = initc

  main(config)