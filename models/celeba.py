# Networks for CelebA dataset. shape: torch.Size([batch_size, 3, 32, 32])

import torch
import torch.nn as nn
import numpy as np

class ParallelNet(nn.Module):
  def forward(self, x):
    gpu_ids = None
    if isinstance(x.data, torch.cuda.FloatTensor):
      gpu_ids = range(2)
    if gpu_ids:
      return nn.parallel.data_parallel(self.main, x, gpu_ids)
    else:
      return self.main(x)

class G(nn.Module):
  def __init__(self):
    super(G, self).__init__()
    
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

class Encoder(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
    super(Encoder, self).__init__()
    self.latent_shape = [hidden_dim, 8, 8]
    layers = []
    layers.append(nn.Conv2d(in_dim, hidden_dim, 3, 1, 1))
    layers.append(nn.ELU(True))

    prev_channel_num = hidden_dim
    for idx in range(repeat_num):
      channel_num = hidden_dim * (idx + 1)
      layers.append(nn.Conv2d(prev_channel_num, channel_num, 3, 1, 1))
      layers.append(nn.ELU(True))
      if idx < repeat_num - 1:
        layers.append(nn.Conv2d(channel_num, channel_num, 3, 2, 1))
      else:
        layers.append(nn.Conv2d(channel_num, channel_num, 3, 1, 1))
      layers.append(nn.ELU(True))
      prev_channel_num = channel_num

    self.conv = nn.Sequential(*layers)
    self.flat_dim = repeat_num*hidden_dim*8*8
    self.fc = nn.Linear(self.flat_dim, out_dim)

  def forward(self, x):
    x = self.conv(x).view(-1, self.flat_dim)
    # print("encoder.out.size:", x.size())
    x = self.fc(x)
    return x
    

class Decoder(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
    super(Decoder, self).__init__()

    self.latent_shape = [hidden_dim, 8, 8]
    self.fc = nn.Linear(in_dim, np.prod(self.latent_shape))
    layers = []
    for idx in range(repeat_num):
      layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1))
      layers.append(nn.ELU(True))
      layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1))
      layers.append(nn.ELU(True))
      if idx < repeat_num - 1:
        layers.append(nn.UpsamplingNearest2d(scale_factor=2))
    layers.append(nn.Conv2d(hidden_dim, out_dim, 3, 1, 1))
    layers.append(nn.ELU(True))
    self.conv = nn.Sequential(*layers)
  
  def forward(self, x):
    x = self.fc(x).view([-1] + self.latent_shape)
    x = self.conv(x)
    # print("decoder.out ", x.size())
    return x

class GeneratorCNN(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
    super(GeneratorCNN, self).__init__()
    self.latent_dim = 64
    self.fc = nn.Linear(in_dim, self.latent_dim)
    self.dec = Decoder(self.latent_dim, out_dim, hidden_dim, repeat_num)

  def forward(self, x):
    x = self.fc(x)
    x = self.dec(x)
    return x

class DiscriminatorCNN(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
    super(DiscriminatorCNN, self).__init__()
    self.latent_dim = 64
    self.enc = Encoder(in_dim, self.latent_dim, hidden_dim, repeat_num)
    self.dec = Decoder(self.latent_dim, out_dim, hidden_dim, repeat_num)

  def forward(self, x):
    latent = self.enc(x)
    x = self.dec(latent)
    return latent, x
    

class Qhead(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Qhead, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.Tanh(),
    )
  
  def forward(self, x):
    x = self.fc(x)
    return x 