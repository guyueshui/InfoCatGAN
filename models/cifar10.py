# Networks for CIFAR10 dataset. shape: torch.Size([batch_size, 3, 32, 32])

import torch.nn as nn

class G(nn.Module):
  def __init__(self, indim=100, outdim=3):
    super(G, self).__init__()
    self.indim = indim
    self.outdim = outdim
    dim = 64

    #
    # Architecture comes from "xinario/catgan_pytorch"
    #
    # in: bs x (indim x 1 x 1)
    self.main = nn.Sequential(
      nn.ConvTranspose2d(self.indim, dim*4, 4, 1, bias=False), # out: dim*4 x 4 x 4
      nn.BatchNorm2d(dim * 4),
      nn.LeakyReLU(0.2, inplace=True),

      nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1, bias=False), # out: dim*2 x 8 x 8
      nn.BatchNorm2d(dim * 2),
      nn.LeakyReLU(0.2, True),

      nn.ConvTranspose2d(dim*2, dim, 4, 2, 1, bias=False), # out: dim x 16 x 16
      nn.BatchNorm2d(dim),
      nn.LeakyReLU(0.2, True),

      nn.ConvTranspose2d(dim, outdim, 4, 2, 1, bias=False), # out: 3 x 32 x 32
      nn.Tanh()
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
    dim = 64
    # in: bs x (3 x 32 x 32)
    self.main = nn.Sequential(
      nn.Conv2d(3, dim, 4, 2, 1, bias=False),  # out: 64 x 16 x 16
      nn.BatchNorm2d(dim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.5),

      nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False),  # out: 128 x 8 x 8
      nn.BatchNorm2d(dim * 2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.5),

      nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=False),  # out: 256 x 4 x 4
      nn.BatchNorm2d(dim * 4),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.5),

      nn.Conv2d(dim*4, dim*4, 4),   # out: 256 x 1 x 1
      nn.BatchNorm2d(dim * 4),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),
    )


  def forward(self, x):
    x = self.main(x)
    return x


class D(nn.Module):
  def __init__(self):
    super(D, self).__init__()

    self.main = nn.Conv2d(256, 10, 1)  # out: 10 x 1 x 1
    self.softmax = nn.Softmax(dim=1)  # Each row sums to 1.

  def forward(self, x):
    x = self.main(x).view(-1, 10)
    x = self.softmax(x)
    return x

class Q(nn.Module):

  def __init__(self):
    super(Q, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(256, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.conv_disc = nn.Conv2d(128, 10, 1)
    self.conv_mu = nn.Conv2d(128, 2, 1)
    self.conv_var = nn.Conv2d(128, 2, 1)

  def forward(self, x):

    y = self.main(x)

    disc_logits = self.conv_disc(y).squeeze()
    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var 

class Qsemi(nn.Module):
  def __init__(self, outdim=10):
    super(Qsemi, self).__init__()
    self.outdim = outdim
    
    self.main = nn.Sequential(
      nn.Conv2d(256, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(128, 10, 1),
    )

  def forward(self, x):
    x = self.main(x).squeeze()
    return x

#============= Architecture for CatGAN ==================#
# see https://github.com/xinario/catgan_pytorch/blob/master/catgan_cifar10.py
nlayers = 64

class CatG(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(CatG, self).__init__()
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(in_dim, nlayers * 4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(nlayers * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (nlayers*8) x 4 x 4
      nn.ConvTranspose2d(nlayers * 4, nlayers * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(nlayers * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (nlayers*4) x 8 x 8
      nn.ConvTranspose2d(nlayers * 2, nlayers, 4, 2, 1, bias=False),
      nn.BatchNorm2d(nlayers),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (nlayers*2) x 16 x 16
      nn.ConvTranspose2d(nlayers, out_dim, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 32 x 32
    )

  def forward(self, x):
    if len(x.size()) == 2:
      a, b = x.size()
      x = x.view(a, b, 1, 1)
    output = self.main(x)
    return output

class CatD(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(CatD, self).__init__()
    self.out_dim = out_dim
    self.main = nn.Sequential(
      nn.Conv2d(in_dim, nlayers, 4, 2, 1, bias=False),
      nn.BatchNorm2d(nlayers),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.5),#64x16x16
      nn.Conv2d(nlayers, 2 * nlayers, 4, 2, 1, bias=False),
      nn.BatchNorm2d(2*nlayers),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.5),#128x8x8
      nn.Conv2d(2 * nlayers, 4 * nlayers, 4, 2, 1, bias=False),
      nn.BatchNorm2d(4*nlayers),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.5),#256x4x4
      nn.Conv2d(4*nlayers, 4*nlayers, 4),
      nn.BatchNorm2d(4*nlayers),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.5),#256x1x1
      nn.Conv2d(4*nlayers, out_dim, 1)
    )

    self.softmax = nn.Softmax(dim=1) # Each row sums to 1.

  def forward(self, x):
    logits = self.main(x).view(-1, self.out_dim)
    simplex = self.softmax(logits)
    return simplex, logits