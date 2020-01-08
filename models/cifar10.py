# Networks for CIFAR10 dataset. shape: torch.Size([batch_size, 3, 32, 32])

import torch.nn as nn

class G(nn.Module):
  def __init__(self, indim=140, outdim=3):
    super(G, self).__init__()
    self.indim = indim
    self.outdim = outdim

    # in: bs x (indim x 1 x 1)
    self.main = nn.Sequential(
      nn.ConvTranspose2d(self.indim, 1024, 1, 1), # out: 1024 x 1 x 1
      nn.BatchNorm2d(1024),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(1024, 128, 8, 1), # out: 128 x 8 x 8
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1), # out: 64 x 16 x 16
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, self.outdim, 4, 2, 1), # out: 3 x 32 x 32
      nn.Sigmoid()
    )
  
  def forward(self, x):
    x = self.main(x)
    return x

class FrontG(nn.Module):
  def __init__(self, indim=74):
    super(FrontG, self).__init__()
    self.indim = indim

    # in: bs x (indim x 1 x 1)
    self.main = nn.Sequential(
      nn.ConvTranspose2d(self.indim, 1024, 1, 1), # out: 1024 x 1 x 1
      nn.BatchNorm2d(1024),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(1024, 128, 8, 1), # out: 128 x 8 x 8
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1), # out: 64 x 16 x 16
      nn.BatchNorm2d(64),
      nn.ReLU(True)
    )
  
  def forward(self, input):
    output = self.main(input)
    return output

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.main = nn.Sequential(
      nn.ConvTranspose2d(64, 3, 4, 2, 1), # out: 3 x 32 x 32
      nn.Sigmoid()
    )
  
  def forward(self, input):
    output = self.main(input)
    return output

class GTransProb(nn.Module):
  'View generator as a channel, output it\'s transition matrix, i.e. p(x|c)'

  def __init__(self):
    super(GTransProb, self).__init__()
    self.main = nn.Conv2d(64, 10, kernel_size=16, bias=False) # batch_size * 10 * 1 * 1
    self.softmax = nn.Softmax(dim=0) # Each column sums to 1.
  
  def forward(self, input):
    output = self.main(input).view(-1, 10)
    output = self.softmax(output)
    output.transpose_(0, 1)
    return output

class FrontD(nn.Module):
  def __init__(self):
    super(FrontD, self).__init__()

    # in: bs x (3 x 32 x 32)
    self.main = nn.Sequential(
      nn.Conv2d(3, 64, 4, 2, 1),  # out: 64 x 16 x 16
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # out: 128 x 8 x 8
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 8, bias=False),  # out: 1024 x 1 x 1
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True)
    )

  def forward(self, input):
    output = self.main(input)
    return output

class D(nn.Module):
  def __init__(self):
    super(D, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1024, 1, 1),  # out: 1 x 1 x 1
      nn.Sigmoid()
    )

  def forward(self, input):
    output = self.main(input).view(-1, 1)
    return output

class Q(nn.Module):

  def __init__(self):
    super(Q, self).__init__()

    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Conv2d(128, 10, 1)
    self.conv_mu = nn.Conv2d(128, 2, 1)
    self.conv_var = nn.Conv2d(128, 2, 1)

  def forward(self, x):

    y = self.conv(x)

    disc_logits = self.conv_disc(y).squeeze()

    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var 

class Qsemi(nn.Module):
  def __init__(self, outdim=10):
    super(Qsemi, self).__init__()
    self.outdim = outdim
    
    self.main = nn.Sequential(
      nn.Conv2d(1024, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 10, 1),
    )

  def forward(self, x):
    x = self.main(x).squeeze()
    return x
