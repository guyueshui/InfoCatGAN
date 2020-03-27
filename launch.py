from IInfoGAN import IInfoGAN
from InfoGAN import InfoGAN
from BEGAN import BEGAN
from SS_InfoGAN import SS_InfoGAN
from config import get_config
import utils

import torch
import numpy as np

def main(config):
  if config.dataset == 'CelebA':
    import models.celeba as nets
    dataset = utils.get_data('CelebA', config.data_root)
    print("This is celeba dataset, " \
          "{} images will be used in training.".format(len(dataset)))
  elif config.dataset == 'MNIST':
    import models.mnist as nets
    dataset = utils.get_data('MNIST', config.data_root)
  else:
    raise NotImplementedError('unsupport dataset')

  if config.gan_type == "ssinfogan":
    gan = SS_InfoGAN(config, dataset)
  elif config.gan_type == "infogan":
    gan = InfoGAN(config, dataset)
  elif config.gan_type == "began":
    gan = BEGAN(config, dataset)
  else:
    raise NotImplementedError('unsupport gan type')
  gan.train()


if __name__ == '__main__':
  args = get_config()
  print(args)
  # Fix random seeds.
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  main(args)

#TODO:
# - reproduce catgan (classification)
# - compute inception score