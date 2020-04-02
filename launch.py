from InfoGAN import InfoGAN
from SS_InfoGAN import SS_InfoGAN
from CatGAN import CatGAN as CatGAN
from config import get_config
import utils

import torch
import numpy as np

def main(config):
  if config.dataset == 'CelebA':
    dataset = utils.get_data('CelebA', config.data_root)
    print("This is celeba dataset, " \
          "{} images will be used in training.".format(len(dataset)))
  elif config.dataset == 'MNIST':
    dataset = utils.get_data('MNIST', config.data_root)
  else:
    raise NotImplementedError('unsupport dataset')

  if config.gan_type == "ssinfogan":
    gan = SS_InfoGAN(config, dataset)
  elif config.gan_type == "infogan":
    gan = InfoGAN(config, dataset)
  elif config.gan_type == "catgan":
    gan = CatGAN(config, dataset)
  else:
    raise NotImplementedError('unsupport gan type')
  gan.train()

  if config.perform_classification:
    from classify import Classifier
    import glob
    
    dataset_to_classify = utils.get_data(config.dataset, config.data_root, train=False)
    for pt in glob.glob(gan.save_dir + '/' + '*.pt'):
      print(pt)
      c = Classifier(gan, pt, dataset_to_classify)
      c.classify()


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
# - add classification part
