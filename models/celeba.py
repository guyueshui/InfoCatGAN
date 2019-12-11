# Networks for CelebA dataset. shape: torch.Size([batch_size, 3, 32, 32])

import torch.nn as nn

class FrontG(nn.Module):
  def __init__(self):
    super(FrontG, self).__init__()
    
    self.main = nn.Sequential(
      nn.ConvTranspose2d(228, 448, 2, 1, bias=False),
      nn.BatchNorm2d(448),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(448, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.ReLU(True),
    )

  def forward(self, input):
    output = self.main(input)
    return output


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
      nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
      nn.Tanh()
    )
  
  def forward(self, input):
    output = self.main(input)
    return output

class FrontD(nn.Module):
  def __init__(self):
    super(FrontD, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(3, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, True),
      nn.Conv2d(128, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.1, True)
    )

  def forward(self, input):
    output = self.main(input)
    return output


class D(nn.Module):
  def __init__(self):
    super(D, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(256, 1, 4),
      nn.Sigmoid()
    )

  def forward(self, input):
    output = self.main(input)
    return output


class Q(nn.Module):
  def __init__(self):
    super(Q, self).__init__()

    self.conv1 = nn.Conv2d(256, 128, 4, bias=False)
    self.bn1 = nn.BatchNorm2d(128)
    self.lReLU1 = nn.LeakyReLU(0.1, inplace=True)

    self.conv_disc = nn.Conv2d(128, 100, 1)

    self.conv_mu = nn.Conv2d(128, 1, 1)
    self.conv_var = nn.Conv2d(128, 1, 1)

  def forward(self, x):
    y = self.lReLU1(self.bn1(self.conv1(x)))

    disc_logits = self.conv_disc(y).squeeze()

    # Not used during training for CelebA.
    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var