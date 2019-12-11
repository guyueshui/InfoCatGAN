import torchvision
import torch
import numpy as np 

from torchvision import transforms
from utils import BlahutArimoto

class Test:
  def ImbalanceTest(self):
    from utils import ImbalanceSampler
    dataset = torchvision.datasets.MNIST('../datasets', transform=transforms.ToTensor())
    sample_probs = np.random.rand(10)
    ib = ImbalanceSampler(dataset, sample_probs)
    ib.Imbalancer()
    ib_ratio = ib.Report()
    print('sample probs:', sample_probs)
    print('ib ratio:', ib_ratio)

  def BATest(self):
    r = np.array([1e-12, 1 - 1e-12])
    pxc = np.array([[0.9, 0.1], [0.5, 0.5]])

########## another method ###############
    maxmi = 0.0
    optim_p = 0
    for p in np.linspace(0.0, 1.0, 1000):
      indist = np.array([p, 1-p])
      mi = MutualInfo(indist, pxc).MI()
      if mi > maxmi:
        maxmi = mi
        print('current mi is ', maxmi)
        optim_p = p
    optim_r = np.array([optim_p, 1 - optim_p])
    print('The optimal input dist is: ', optim_r)
################ end ####################
      

    for i in range(pxc.shape[0]):
      factor = np.sum(pxc[i,:])
      for j in range(pxc.shape[1]):
        pxc[i][j] /= factor
    print('r is {}, pxc is {}'.format(r, pxc))

    ba = BlahutArimoto(r, pxc)
    optim_r, post_dist = ba.Update(1e-6)
    print('optim r is {}, post dist is {}'.format(optim_r, post_dist))
    print('shapes: ', optim_r.shape, post_dist.shape)
    print('sum of r is {}, of post dist is {}'.format(np.sum(optim_r), np.sum(post_dist, 1)))
    return optim_r

class MutualInfo:
  'Compute the mutual infomation of the given input distribution and channel transition matrix.'
  def __init__(self, input_dist, trans_mat):
    self.input_dist = input_dist.reshape(1, -1)
    self.trans_mat = trans_mat

  def entropy(self, dist: np.ndarray):
    vec = [- p * np.log2(p) for p in dist] 
    return np.sum(vec)

  def MI(self):
    PY = np.matmul(self.input_dist, self.trans_mat)
    HY_x = [self.entropy(self.trans_mat[i,:]) for i in range(self.trans_mat.shape[0])]
    HY_X = np.sum(self.input_dist * HY_x)
    HY = self.entropy(PY)
    return HY - HY_X
    

if __name__ == '__main__':
  t = Test()
  # t.ImbalanceTest()
  t.BATest()
