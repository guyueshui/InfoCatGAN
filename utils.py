import torch
import numpy as np
import copy

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

    assert(r.size == channel.shape[0])
    assert(abs(np.sum(r) - 1) < gTiny)
    assert((abs(
      np.sum(channel, axis=1) - np.ones(channel.shape[0])
      ) < gTiny).all())

    self.input_dist = r                 # r(c): (1, 10)
    self.transition = channel           # p(x|c): (10, 128)
    # self.post_dist = np.zeros(shape=(   # q(c|x): (128, 10)
    #   channel.shape[1], channel.shape[0]
    # ))
    self.num_cat = r.size
    self.batch_size = channel.shape[1]

  def Update(self, epsilon: float):
    from queue import Queue
    que = Queue(maxsize=2)
    que.put(0)
    while True:
      cur_f, post_dist = self.Objective()
      que.put(cur_f)
      prev_f = que.get()
      if abs(cur_f - prev_f) < epsilon:
        break
    return self.input_dist, post_dist
    
  def Objective(self):
    # update q(c|x)
    post_dist = np.asarray([
      (self.input_dist * self.transition[:,i]).squeeze()
      for i in range(self.transition.shape[1])
    ]).reshape(self.batch_size, self.num_cat)

    for i in range(post_dist.shape[0]): # normalize
      rowsum = np.sum(post_dist[i,:])
      for j in range(post_dist.shape[1]):
        post_dist[i][j] /= rowsum

    # compute objective
    f = np.sum(
      np.matmul(
        np.matmul(self.input_dist, self.transition),
        np.log2(post_dist / self.input_dist)
      )
    )

    # update r(c)
    logrc = np.asarray([
      np.matmul(self.transition[i,:], post_dist[:,i])
      for i in range(self.num_cat)
    ]).reshape(1, -1)
    self.input_dist = np.exp2(logrc)
    self.input_dist = self.input_dist / np.sum(self.input_dist)

    return f, post_dist

class ImbalanceSampler:
  def __init__(self, dataset, sample_probs):
    self.imbalanced_dataset = copy.deepcopy(dataset)

    # Resample the dataset to imbalanced dataset according to @sample_probs.
    idx_to_del = [i for i, label in enumerate(dataset.targets)
                  if np.random.rand() > sample_probs[label]]
    self.imbalanced_dataset.targets = np.delete(dataset.targets, idx_to_del, axis=0)
    self.imbalanced_dataset.data = np.delete(dataset.data, idx_to_del, axis=0)

  def ImbalancedDataset(self):
    'Report the imbalanced class distribution of the dataset.'
    label, counts = np.unique(self.imbalanced_dataset.targets, return_counts=True)
    imbalanced_dist = counts / np.sum(counts)
    return self.imbalanced_dataset, imbalanced_dist

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)

gTiny = 1e-6
class LogGaussian:
  'Custom loss for Q network.'
  def __call__(self, x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor):
    logli = -0.5 * (var.mul(2*np.pi) + gTiny).log() - \
            (x-mu).pow(2).div(var.mul(2.0) + gTiny)
    return logli.sum(1).mean().mul(-1)