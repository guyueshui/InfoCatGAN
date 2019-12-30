# Architecture from InfoGAN paper.

import torch.nn as nn

class G(nn.Module):
  'G = FrontG + Generator'
  def __init__(self, in_dim=74, out_dim=1):
    super(G, self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim

    # bs x input_dim
    self.fc = nn.Sequential(
      nn.Linear(self.in_dim, 1024),  # bs x 1024
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Linear(1024, 128 * 7 * 7), # bs x 128*7*7
      nn.BatchNorm1d(128 * 7 * 7),
      nn.ReLU()
    )

    # torch.Size([bs, 128, 7, 7])
    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(128, 64, 4, 2, 1), # bs x 64 x 14 x 14
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, self.out_dim, 4, 2, 1), # bs x output_dim x 28 x 28
      nn.Tanh()
    )
    # torch.Size([bs, 1, 28, 28])

  def forward(self, x):
    x = x.view(-1, 74)
    x = self.fc(x).view(-1, 128, 7, 7)
    x = self.deconv(x)
    return x

class FrontG(nn.Module):
  def __init__(self, input_dim=74):
    super(FrontG, self).__init__()
    self.input_dim = input_dim

    # bs x input_dim
    self.fc = nn.Sequential(
      nn.Linear(self.input_dim, 1024),  # bs x 1024
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Linear(1024, 128 * 7 * 7), # bs x 128*7*7
      nn.BatchNorm1d(128 * 7 * 7),
      nn.ReLU()
    )

    # bs x 128 x 7 x 7
    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(128, 64, 4, 2, 1), # bs x 64 x 14 x 14
      nn.BatchNorm2d(64),
      nn.ReLU(),
    )

  def forward(self, x):
    x = x.view(-1, 74)
    x = self.fc(x).view(-1, 128, 7, 7)
    x = self.deconv(x)
    return x


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    # torch.Size([bs, 64, 14, 14])
    self.main = nn.Sequential(
      nn.ConvTranspose2d(64, 1, 4, 2, 1), # bs x output_dim x 28 x 28
      nn.Tanh()
    )
    # torch.Size([bs, 1, 28, 28])

  def forward(self, x):
    x = self.main(x)
    return x


class GTransProb(nn.Module):
  'View generator as a channel, output it\'s transition matrix, i.e. p(x|c)'

  def __init__(self):
    super(GTransProb, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(64*14*14, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 10),
    )
    # self.fc2 = nn.Linear(100, 10)
    self.softmax = nn.Softmax(dim=1) # Each row sums to 1.
  
  def forward(self, x):
    x = x.view(-1, 64*14*14)
    x = self.fc(x)          # bs x 10
    x = x.transpose(1, 0)   # 10 x bs
    x = self.softmax(x)
    return x


class FrontD(nn.Module):
  def __init__(self, in_dim=1, out_dim=1):
    super(FrontD, self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim

    # in_dim x 28 x 28
    self.conv = nn.Sequential(
      nn.Conv2d(self.in_dim, 64, 4, 2, 1),  # 64 x 14 x 14
      nn.LeakyReLU(0.2),
      nn.Conv2d(64, 128, 4, 2, 1),  # 128 x 7 x 7
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2)
    )

    self.fc = nn.Sequential(
      nn.Linear(128*7*7, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    x = self.conv(x).view(-1, 128*7*7)
    x = self.fc(x)  # bs x 1024
    return x


class D(nn.Module):
  def __init__(self):
    super(D, self).__init__()
    self.main = nn.Sequential(
      nn.Linear(1024, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.main(x)
    return x


class Q(nn.Module):
  def __init__(self, dis_dim=10, con_dim=2):
    super(Q, self).__init__()
    self.dis_dim = dis_dim
    self.con_dim = con_dim
    
    self.fc = nn.Sequential(
      nn.Linear(1024, 128),
      nn.BatchNorm1d(128),
      nn.LeakyReLU(0.2),
      # nn.Linear(128, self.dis_dim + self.con_dim)
    )

    self.disc = nn.Linear(128, dis_dim)
    self.mu = nn.Linear(128, con_dim)
    self.var = nn.Linear(128, con_dim)

  def forward(self, x):
    x = self.fc(x)
    disc_logits = self.disc(x)
    mu = self.mu(x)
    var = self.var(x).exp()
    return disc_logits, mu, var


class Qsemi(nn.Module):
  def __init__(self, out_dim=10):
    super(Qsemi, self).__init__()

    self.fc = nn.Linear(1024, out_dim)

  def forward(self, x):
    disc_logits = self.fc(x)
    return disc_logits