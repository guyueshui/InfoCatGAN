# Networks for MNIST dataset. torch.Size([batch_size, 1, 28, 28])

import torch.nn as nn

#========= following arch from =============
# https://github.com/pianomania/infoGAN-pytorch/blob/master/model.py
class FrontD(nn.Module):
  ''' front end part of discriminator and Q'''
  def __init__(self, in_dim=1, out_dim=1024):
    super(FrontD, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(in_dim, 64, 4, 2, 1),           # 28x28 -> 14x14
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 14x14 -> 7x7
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, out_dim, 7, bias=False),      # 7x7 -> 1x1
      nn.BatchNorm2d(out_dim),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    output = self.main(x)
    return output


class D(nn.Module):
  def __init__(self, in_dim=1024, out_dim=1):
    super(D, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(in_dim, out_dim, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.main(x).view(-1, 1)
    return output


class Q(nn.Module):
  def __init__(self, in_dim=1024, cat_dim=10, num_cont_code=2):
    super(Q, self).__init__()
    self.conv = nn.Conv2d(in_dim, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Conv2d(128, cat_dim, 1)
    self.conv_mu = nn.Conv2d(128, num_cont_code, 1)
    self.conv_var = nn.Conv2d(128, num_cont_code, 1)

  def forward(self, x):
    y = self.lReLU(self.bn(self.conv(x)))

    disc_logits = self.conv_disc(y).squeeze()
    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()
    return disc_logits, mu, var 


class G(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(G, self).__init__()
    self.in_dim = in_dim
    self.main = nn.Sequential(
      nn.ConvTranspose2d(in_dim, 1024, 1, 1, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, out_dim, 4, 2, 1, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = x.view(-1, self.in_dim, 1, 1)
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

class GHead(nn.Module):
  def __init__(self):
    super(GHead, self).__init__()

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
  def __init__(self, in_dim, out_dim):
    super(Discriminator, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(in_dim, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(1024, out_dim, 1),
    )

    self.softmax = nn.Softmax(dim=1)  # Each row sums to 1.
  
  def forward(self, input):
    x = self.main(input).squeeze()
    x = self.softmax(x)
    return x


#====== following arch from InfoGAN paper =======
class OfficialGenerator(nn.Module):
  'Arch from InfoGAN paper.'
  def __init__(self, in_dim=74, out_dim=1):
    super(OfficialGenerator, self).__init__()

    # bs x input_dim
    self.fc = nn.Sequential(
      nn.Linear(in_dim, 1024),  # bs x 1024
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
      nn.ConvTranspose2d(64, out_dim, 4, 2, 1), # bs x output_dim x 28 x 28
      nn.Tanh()
    )
    # torch.Size([bs, 1, 28, 28])

  def forward(self, x):
    x = self.fc(x).view(-1, 128, 7, 7)
    x = self.deconv(x)
    return x


class OfficialDbody(nn.Module):
  'Arch from InfoGAN paper.'
  def __init__(self, in_dim, out_dim=1024):
    super(OfficialDbody, self).__init__()

    # in_dim x 28 x 28
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 64, 4, 2, 1),  # 64 x 14 x 14
      nn.LeakyReLU(0.2),
      nn.Conv2d(64, 128, 4, 2, 1),  # 128 x 7 x 7
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2)
    )

    self.fc = nn.Sequential(
      nn.Linear(128*7*7, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2),
    )

  def forward(self, x):
    x = self.conv(x).view(-1, 128*7*7)
    x = self.fc(x)
    return x


class OfficialDhead(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(OfficialDhead, self).__init__()
    self.main = nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.main(x)
    return x


class OfficialQ(nn.Module):
  def __init__(self, latent_dim, dis_dim=10, con_dim=2):
    super(OfficialQ, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(latent_dim, 128),
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


class OfficialCatD(nn.Module):
  'Arch from InfoGAN paper.'
  def __init__(self, in_dim, out_dim):
    super(OfficialCatD, self).__init__()

    # in_dim x 28 x 28
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 64, 4, 2, 1),  # 64 x 14 x 14
      nn.LeakyReLU(0.2),
      nn.Conv2d(64, 128, 4, 2, 1),  # 128 x 7 x 7
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2)
    )

    self.fc = nn.Sequential(
      nn.Linear(128*7*7, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, out_dim),
    )

    self.softmax = nn.Softmax(dim=1) # Each row sums to 1.

  def forward(self, x):
    x = self.conv(x).view(-1, 128*7*7)
    logits = self.fc(x)
    simplex = self.softmax(logits)
    return simplex, logits

#======== following arch from CatGAN paper? ========
# but fails to drive training process.
class CatD(nn.Module):
  "Arch from CatGAN paper."
  def __init__(self, in_dim, out_dim):
    super(CatD, self).__init__()

    # in_dim x 28 x 28
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 32, 5),  # 24 x 24
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(3, 2),  # 11 x 11
      nn.Conv2d(32, 64, 3), # 9 x 9
      nn.LeakyReLU(0.1),
      nn.Conv2d(64, 64, 3), # 7 x 7
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(3, 2),  # 3 x 3
      nn.Conv2d(64, 128, 3), # 1 x 1
      nn.LeakyReLU(0.1),
      nn.Conv2d(128, 128, 1),  # 1 x 1
      nn.LeakyReLU(0.1),
    )

    self.fc = nn.Sequential(
      nn.Linear(128, out_dim),
      nn.LeakyReLU(0.1),
      # nn.Softmax(dim=1),
    )

    self.softmax = nn.Softmax(dim=1) # Each row sums to 1.

  def forward(self, x):
    x = self.conv(x).view(-1, 128)
    # print("after conv ", x.size())
    logits = self.fc(x)
    simplex = self.softmax(logits)
    return simplex, logits


class CatG(nn.Module):
  'Architecture from CatGAN paper.'
  def __init__(self, in_dim=128, out_dim=1):
    super(CatG, self).__init__()

    # bs x input_dim
    self.fc = nn.Sequential(
      nn.Linear(in_dim, 7*7*96),
      nn.BatchNorm1d(7*7*96),
      nn.LeakyReLU(0.1),
    )

    self.deconv = nn.Sequential(
      # nn.MaxUnpool2d(2),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(96, 64, 5, 1, 2),
      nn.LeakyReLU(0.1),

      # nn.MaxUnpool2d(2),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 64, 5, 1, 2),
      nn.LeakyReLU(0.1),

      nn.Conv2d(64, out_dim, 5, 1, 2),
      nn.LeakyReLU(0.1),
    )

  def forward(self, x):
    x = self.fc(x).view(-1, 96, 7, 7)
    x = self.deconv(x)
    return x


#========= following arch from tripleGAN paper ============
import torch
from torch.nn.utils import weight_norm

class GaussianNoise(nn.Module):
  """Gaussian noise regularizer.

  Args:
      sigma (float, optional): relative standard deviation used to generate the
          noise. Relative means that it will be multiplied by the magnitude of
          the value your are adding the noise to. This means that sigma can be
          the same regardless of the scale of the vector.
      is_relative_detach (bool, optional): whether to detach the variable before
          computing the scale of the noise. If `False` then the scale of the noise
          won't be seen as a constant but something to optimize: this will bias the
          network to generate vectors with smaller values.
  """

  def __init__(self, sigma=0.1, is_relative_detach=True):
    super().__init__()
    self.sigma = sigma
    self.is_relative_detach = is_relative_detach
    #self.noise = torch.tensor(0).to(device)
    self.noise = torch.tensor(0)

  def forward(self, x):
    if self.training and self.sigma != 0:
      scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
      sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
      x = x + sampled_noise
    return x 

class tg_Generator(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(tg_Generator, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(in_dim, 500),
      nn.BatchNorm1d(500),
      nn.Softplus(),
      nn.Linear(500, 500),
      nn.BatchNorm1d(500),
      nn.Softplus(),
      nn.Linear(500, 28*28),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.fc(x).view(-1, 1, 28, 28)
    return x

class tg_Discriminator(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(tg_Discriminator, self).__init__()
    self.fc = nn.Sequential(
      GaussianNoise(),
      weight_norm( nn.Linear(in_dim, 1000), name='weight'),
      nn.LeakyReLU(0.1),
      GaussianNoise(),
      weight_norm( nn.Linear(1000, 500), name='weight'),
      nn.LeakyReLU(0.1),
      GaussianNoise(),
      weight_norm( nn.Linear(500, 250), name='weight'),
      nn.LeakyReLU(0.1),
      GaussianNoise(),
      weight_norm( nn.Linear(250, 250), name='weight'),
      nn.LeakyReLU(0.1),
      GaussianNoise(),
      weight_norm( nn.Linear(250, 250), name='weight'),
      nn.LeakyReLU(0.1),
      GaussianNoise(),
      weight_norm( nn.Linear(250, out_dim), name='weight'),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.fc(x)
    return x


class tg_Classifier(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(tg_Classifier, self).__init__()
    # torch.Size([bs, 1, 28, 28])
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 32, 5), # 32 x 24 x 24
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2, 2), # 32 x 12 x 12
      nn.Dropout2d(0.5),

      nn.Conv2d(32, 64, 3), # 64 x 10 x 10
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.Conv2d(64, 64, 3), # 64 x 8 x 8
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2, 2), # 64 x 4 x 4
      nn.Dropout2d(0.5),

      nn.Conv2d(64, 128, 3),  # 128 x 2 x 2
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.Conv2d(128, out_dim, 2),  # 10 x 1 x 1
      nn.Softmax(dim=1),  # each row sums to 1.
    )

  def forward(self, x):
    x = self.conv(x)
    return x