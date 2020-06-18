# Hyper parameters.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=70)
#parser.add_argument('--num_class', type=int, default=10, help='Number of different categories.')
#parser.add_argument('--num_noise_dim', type=int, default=62, help='Uncompressed noise dimension (i.e., dim of z) of the generator input.')
parser.add_argument('--data_root', type=str, default='../datasets')
parser.add_argument('--gpu', type=int, default=0, help='[-1 | 0 | 1], which GPU to use, -1 indicates CPU.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_epoch', type=int, default=25, help='After how many epochs to save the checkpoint.')
#parser.add_argument('--num_dis_c', type=int, default=1, help='Number of categorical codes.')
#parser.add_argument('--num_con_c', type=int, default=2, help='Number of continous latent codes.')
parser.add_argument('-f', help='For ipython debug.')
parser.add_argument('--instance_noise', action='store_true', help='Whether to use instance noise trick.')
#parser.add_argument('--fid', action='store_true', help='Whether to compute FID score.')
parser.add_argument('--nlabeled', type=int, default=100)
parser.add_argument('--dbname', type=str, default='MNIST')


def GetConfig():
    config = parser.parse_args()
    return config
