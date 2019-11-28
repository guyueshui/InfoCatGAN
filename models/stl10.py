# Networks for STL10 dataset. shape: torch.Size([batch_size, 3, 96, 96])

import torch.nn as nn

class FrontG(nn.Module):
  def __init__(self):
    super(FrontG, self).__init__()

    # in: bs x (74 x 1 x 1)
    self.main = nn.Sequential(
      nn.ConvTranspose2d(74, 1024, 1, 1), # out: 1024 x 1 x 1
      nn.BatchNorm2d(1024),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(1024, 128, 12, 1), # out: 128 x 12 x 12
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1), # out: 64 x 24 x 24
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 32, 4, 2, 1), # out: 32 x 48 x 48
      nn.BatchNorm2d(32),
      nn.ReLU(True),
    )
  
  def forward(self, input):
    output = self.main(input)
    return output

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.main = nn.Sequential(
      nn.ConvTranspose2d(32, 3, 4, 2, 1), # out: 3 x 96 x 96
      nn.Sigmoid()
    )
  
  def forward(self, input):
    output = self.main(input)
    return output

class GTransProb(nn.Module):
  'View generator as a channel, output it\'s transition matrix, i.e. p(x|c)'

  def __init__(self):
    super(GTransProb, self).__init__()
    self.main = nn.Conv2d(32, 10, kernel_size=48, bias=False) # batch_size * 10 * 1 * 1
    self.softmax = nn.Softmax(dim=0) # Each column sums to 1.
  
  def forward(self, input):
    output = self.main(input).view(-1, 10)
    output = self.softmax(output)
    output.transpose_(0, 1)
    return output

class FrontD(nn.Module):
  def __init__(self):
    super(FrontD, self).__init__()

    # in: bs x (3 x 96 x 96)
    self.main = nn.Sequential(
      nn.Conv2d(3, 32, 4, 2, 1),  # out: 32 x 48 x 48
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # out: 64 x 24 x 24
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # out: 128 x 12 x 12
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 12, bias=False),  # out: 1024 x 1 x 1
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