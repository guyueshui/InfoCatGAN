# Networks for MNIST dataset. torch.Size([batch_size, 1, 28, 28])

import torch.nn as nn

class FrontD(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(FrontD, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),                # 28x28 -> 14x14
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 14x14 -> 7x7
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),      # 7x7 -> 1x1
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    output = self.main(x)
    return output


class D(nn.Module):

  def __init__(self):
    super(D, self).__init__()
    
    self.main = nn.Sequential(
      nn.Conv2d(1024, 1, 1),
      nn.Sigmoid()
    )
    

  def forward(self, x):
    output = self.main(x).view(-1, 1)
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

    y = self.lReLU(self.bn(self.conv(x)))

    disc_logits = self.conv_disc(y).squeeze()

    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var 


class G(nn.Module):

  def __init__(self):
    super(G, self).__init__()

    self.main = nn.Sequential(
      nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.main(x)
    return output

class FrontG(nn.Module):
  def __init__(self):
    super(FrontG, self).__init__()
  
    # torch.Size([bs, 74, 1, 1])
    self.main = nn.Sequential(
      nn.ConvTranspose2d(74, 1024, 1, 1, bias=False), # 1x1 -> 1x1
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),  # 1x1 -> 7x7
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # 7x7 -> 14x14
      nn.BatchNorm2d(64),
      nn.ReLU(True)
    ) # torch.Size([bs, 64, 14, 14])

  def forward(self, input):
    output = self.main(input)
    return output

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    # torch.Size([bs, 64, 14, 14])
    self.main = nn.Sequential(
      nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14 -> 28x28
      nn.Sigmoid()
    )
    # torch.Size([bs, 1, 28, 28])
  
  def forward(self, input):
    output = self.main(input)
    return output

class GTransProb(nn.Module):
  'View generator as a channel, output it\'s transition matrix, i.e. p(x|c)'

  def __init__(self):
    super(GTransProb, self).__init__()
    self.conv1 = nn.Conv2d(64, 10, kernel_size=14, bias=False) # batch_size * 10 * 1 * 1
    self.bn1 = nn.BatchNorm2d(10)
    self.lReLU1 = nn.LeakyReLU(0.1, inplace=True)
    self.fc1 = nn.Linear(100, 10, bias=True)
    self.softmax = nn.Softmax(dim=1) # Each row sums to 1.
  
  def forward(self, input):
    output = self.lReLU1(self.bn1(self.conv1(input))).view(10, -1)
    output = self.fc1(output)
    output = self.softmax(output)
    return output

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(1024, 1, 1),
      nn.Sigmoid()
    )
  
  def forward(self, input):
    output = self.main(input).view(-1, 1)
    return output