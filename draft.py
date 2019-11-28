import torchvision
import torch
import numpy as np 

from torchvision import transforms

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

if __name__ == '__main__':
  t = Test()
  t.ImbalanceTest()
