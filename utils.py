import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import copy
import matplotlib.pyplot as plt
import time as t
import imageio
import math

from torchvision.datasets import VisionDataset

class Noiser:
  'Generate some noise for GAN\'s input.'

  @staticmethod
  def Category(prob: np.ndarray, batch_size: int): 
    'Generate onehot samples from given prob.'
    idx = torch.multinomial(
      torch.Tensor(prob), batch_size, replacement=True)
    idx = idx.numpy()
    onehot = np.zeros(shape=(batch_size, len(prob)))
    onehot[range(batch_size), idx] = 1.0
    return onehot, idx
  
  @staticmethod
  def Uniform(lower: float, upper: float, rows, cols=1):
    'Generate continous uniform samples of given size.'
    t = torch.FloatTensor(rows, cols)
    t.uniform_(lower, upper)
    return t.numpy()

  @staticmethod
  def Normal(mean: float, std: float, rows, cols=1):
    'Generate continous normal samples of given size.'
    t = troch.FloatTensor(rows, cols)
    t.normal_(mean, std)
    return t.numpy()


class BlahutArimoto:
  """
  The BA algorithm for computing optimal distribution of c1.

  The objective is:
    f(r, q) = sum_c sum_x [r(c) * p(x|c) * log( q(c|x) / r(c) )]

  What if p(x|c), c -> Generator -> x.
  We feed noise c into generator, and get fake example x. The cardinality
  of x is set to batch_size.
  """
  def __init__(self, r: np.ndarray, channel: np.ndarray):

    # assert(r.size == channel.shape[0])
    # assert(abs(np.sum(r) - 1) < gTiny)
    # assert((abs(
    #   np.sum(channel, axis=1) - np.ones(channel.shape[0])
    #   ) < gTiny).all())

    # Ensure "self.input_dist > self.tiny" element-wise.
    self.input_dist = np.clip(r, 1e-3, 1.0)        # r(c): (1, 10)
    self.transition = channel           # p(x|c): (10, 128)
    # self.post_dist                    # q(c|x): (128, 10)

    self.in_channels = r.size
    self.out_channels = channel.shape[1]
    self.input_dist /= np.sum(self.input_dist)

  def Update(self, epsilon: float):
    """
    Note
      epsilon should be at least 2 level less than the minimum of the
      input dist (self.input_dist). Otherwise, BA algo will collapse.
      
      E.g., if min(self.input_dist) = 1e-17, then set 'epsilon < 1e-19'.
    """
    from queue import Queue
    que = Queue(maxsize=2)
    que.put(0)
    # print('==> Updating...')
    num_iter = 0
    while True:
      cur_f, post_dist = self.Objective()
      que.put(cur_f)
      # print('cur objective is {}'.format(cur_f))
      prev_f = que.get()

      if cur_f < prev_f or cur_f < 0: # bad condition
        print('cur input_dist: {:}, cur_f: {:>.6f}'.format(self.input_dist, cur_f))

      if num_iter > 200 or abs(cur_f - prev_f) < epsilon:
        break
      num_iter += 1
    
    return self.input_dist, post_dist
    
  def Objective(self):
    # update q(c|x)
    post_dist = np.asarray([
      (self.input_dist * self.transition[:,i]).squeeze()
      for i in range(self.out_channels)
    ]).reshape(self.out_channels, self.in_channels)

    factor = np.sum(post_dist, axis=1).reshape(-1, 1)
    post_dist /= factor # normalize

    # compute objective
    f = 0.0
    for c in range(self.in_channels):
      sumx = self.input_dist[c] * self.transition[c,:]
      sumx *= np.log2(post_dist[:,c] / (self.input_dist[c] + 0.0))
      f += np.sum(sumx)
      # Or expand it to a loop.
      # for x in range(self.out_channels):
      #   f += (self.input_dist[c] * self.transition[c, x] * 
      #         np.log2(post_dist[x, c] / self.input_dist[c]))
    # assert(f > 0)

    # update r(c)
    logrc = np.asarray([
      np.sum(self.transition[c,:] * np.log2(post_dist[:,c]))
      for c in range(self.in_channels)
    ])
    self.input_dist = np.exp2(logrc)
    self.input_dist /= np.sum(self.input_dist) # normalize

    return f, post_dist


class ImbalanceSampler:
  'Make an imbalanced dataset by deleting something.'

  def __init__(self, dataset, sample_probs):
    self.imbalanced_dataset = copy.deepcopy(dataset)

    # Resample the dataset to imbalanced dataset according to @sample_probs.
    idx_to_del = [i for i, label in enumerate(dataset.targets)
                  if np.random.rand() > sample_probs[label]]
    self.imbalanced_dataset.targets = np.delete(dataset.targets, idx_to_del, axis=0)
    self.imbalanced_dataset.data = np.delete(dataset.data, idx_to_del, axis=0)

  def ImbalancedDataset(self):
    'Return the imbalanced dataset and it\'s distribution.'
    label, counts = np.unique(self.imbalanced_dataset.targets, return_counts=True)
    imbalanced_dist = counts / np.sum(counts)
    return self.imbalanced_dataset, imbalanced_dist


class LogGaussian:
  """
  Calculate the negative log likelihood of normal distribution.
  Treat Q(c|x) as a factored Gaussian.
  Custom loss for Q network.
  """
  def __call__(self, x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor):
    logli = -0.5 * (var.mul(2*np.pi) + 1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0) + 1e-6)
    return logli.sum(1).mean().mul(-1)


class ETimer:
  'A easy-to-use timer.'

  def __init__(self):
    self._start = t.time()

  def reset(self):
    self._start = t.time()

  def elapsed(self):  # In seconds.
    return t.time() - self._start


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)
  elif classname.find('Linear') != -1:
    m.weight.data.normal_(0, 0.02)
    m.bias.data.fill_(0)


def get_data(dbname: str, data_root: str):
  'Get training dataset.'

  if dbname == 'MNIST':
    transform = transforms.Compose([
      transforms.Resize(28),
      transforms.CenterCrop(28),
      transforms.ToTensor()
    ])

    dataset = dsets.MNIST(data_root, train=True, download=True, transform=transform)

  elif dbname == 'FashionMNIST':
    transform = transforms.Compose([
      transforms.Resize(28),
      transforms.CenterCrop(28),
      transforms.ToTensor()
    ])

    dataset = dsets.FashionMNIST(data_root, train=True, transform=transform, 
                                 download=True)

  elif dbname == "CIFAR10":
    transform = transforms.Compose([
      transforms.Resize(32),
      transforms.CenterCrop(32),
      transforms.ToTensor(),
    ])
    
    dataset = dsets.CIFAR10(data_root, train=True, transform=transform, 
                            download=True)
    
  elif dbname == 'CelebA':
    transform = transforms.Compose([
      transforms.CenterCrop(160),
      transforms.Resize(64),
      transforms.ToTensor(),
    ])

    dataset = dsets.CelebA(data_root, transform=transform, download=False)
    # dataset = dsets.ImageFolder(data_root, transform=transform)

  elif dbname == 'STL10':
    transform = transforms.Compose([
      transforms.Resize(96),
      transforms.CenterCrop(96),
      transforms.ToTensor(),
    ])

    dataset = dsets.STL10(data_root, transform=transform)
  
  else:
    raise NotImplementedError

  return dataset


def generate_animation(path: str, images: list):
  imageio.mimsave(path + '/' + 'generate_animation.gif', images, fps=5)


# Add custom dataset for semisupervised training.
class CustomDataset:
  "Split the origin dataset into labeled and unlabeled by the given ratio."
  def __init__(self, dset, supervised_ratio):
    self.num_data = len(dset)
    self.num_labeled_data = math.ceil(len(dset) * supervised_ratio)
    self.num_unlabeled_data = self.num_data - self.num_labeled_data

    # Make a uniform labeled subset.
    num_classes = len(dset.class_to_idx)
    num_data_per_class = self.num_labeled_data // num_classes
    targets_to_draw = np.arange(num_classes).repeat(num_data_per_class).tolist()
    while len(targets_to_draw) < self.num_labeled_data:
      targets_to_draw.append(np.random.randint(num_classes))
    idx_to_draw = []
    start = 0
    for i, label in enumerate(dset.targets):
      if start < len(targets_to_draw):
        if label == targets_to_draw[start]:
          idx_to_draw.append(i)
          start += 1
      else:
        break
    mask = np.zeros(len(dset), dtype=bool)
    mask[idx_to_draw] = True
    _, cnts = np.unique(targets_to_draw, return_counts=True)
    self.labeled_dist = cnts / np.sum(cnts)

    self.labeled_data = copy.deepcopy(dset)
    self.labeled_data.data = dset.data[mask]
    self.labeled_data.targets = np.asarray(dset.targets)[mask].tolist()

    self.unlabeled_data = copy.deepcopy(dset)
    self.unlabeled_data.data = np.delete(dset.data, idx_to_draw, axis=0)
  
  @property
  def labeled(self):
    return self.labeled_data

  @property
  def unlabeled(self):
    return self.unlabeled_data

  def report(self):
    print('-'*25)
    print('Origin dataset has {} samples'.format(self.num_data))
    print('Now splitted into {} labeled/ {} unlabeled'
          .format(len(self.labeled_data), len(self.unlabeled_data)))
    print('The labeled dist is: ', self.labeled_dist)
    print('-'*25)
    

def MarginalEntropy(y):
  y1 = torch.autograd.Variable(torch.randn(y.size(1)).type(torch.FloatTensor), requires_grad=True)
  y2 = torch.autograd.Variable(torch.randn(1).type(torch.FloatTensor), requires_grad=True)
  y1 = y.mean(0)
  y2 = -torch.sum(y1 * torch.log(y1 + 1e-6))
  return y2

def Entropy(y):
  bs = y.size(0)
  y1 = torch.autograd.Variable(torch.randn(y.size()).type(torch.FloatTensor), requires_grad=True)
  y2 = torch.autograd.Variable(torch.randn(1).type(torch.FloatTensor), requires_grad=True)
  y1 = -y * torch.log(y + 1e-6)
  y2 = 1.0 / bs * y1.sum()
  return y2

def DrawDistribution(dataset, title='Distribution of dataset'):
  'Draw the distribution per label of a given dataset.'
  label, counts = np.unique(dataset.targets, return_counts=True)
  fig, ax = plt.subplots()
  ax.bar(label, counts)
  ax.set_xticks(label)
  ax.set_title(title)
  fig.show()
