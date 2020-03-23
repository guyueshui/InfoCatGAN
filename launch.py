from IInfoGAN import IInfoGAN
from InfoGAN import InfoGAN
from BEGAN import BEGAN
from config import get_config
import utils

import torch

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
    raise NotImplementedError

  gan = InfoGAN(config, dataset)
  gan.train()


if __name__ == '__main__':
  args = get_config()
  print(args)
  main(args)