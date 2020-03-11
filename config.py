# Hyper parameters.

import argparse
import numpy as np

def str2bool(v):
  return v.lower() in ['True', '1']

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=70)
parser.add_argument('--num_class', type=int, default=10, help='Number of different categories.')
parser.add_argument('--num_noise_dim', type=int, default=128, help='Uncompressed noise dimension (i.e., dim of z) of the generator input.')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--tiny', type=float, default=1e-6, help='Global precison.')
parser.add_argument('--dataset', type=str, default='MNIST', help='[MNIST | FashionMNIST | CelebA], The dataset to train.')
parser.add_argument('--data_root', type=str, default='../datasets')
parser.add_argument('--gpu', type=int, default=0, help='[-1 | 0 | 1], which GPU to use, -1 indicates CPU.')
parser.add_argument('--device', default=None, help='Delayed determined by gpu.')
# parser.add_argument('--use_ba', action='store_true', help='Wether to use BlahutArimoto algo.')
parser.add_argument('--imbalanced', action='store_true', help='Wether to process the dataset to be imbalanced.')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproduction.')
parser.add_argument('--experiment_tag', type=str, default='unifc', help='Experiment tag that gives some meaningful info.')
parser.add_argument('--save_epoch', type=int, default=25, help='After how many epochs to save the checkpoint.')
parser.add_argument('--num_dis_c', type=int, default=1, help='Number of categorical codes.')
parser.add_argument('--num_con_c', type=int, default=2, help='Number of continous latent codes.')
parser.add_argument('-f', help='For ipython debug.')
parser.add_argument('--instance_noise', type=str2bool, default=False, help='Whether to use instance noise trick.')

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lambda_k', type=float, default=0.001)
parser.add_argument('--hidden_dim', type=int, default=128)

def get_config():
  config = parser.parse_args()
  return config
