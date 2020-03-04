# Networks for CelebA dataset. shape: torch.Size([batch_size, 3, 32, 32])

import torch
import torch.nn as nn
import numpy as np

dv = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
  def __init__(self, in_shape, out_shape, repeat_num):
    super(Encoder, self).__init__()
    self.out_shape = out_shape
    in_channel, in_h, in_w = in_shape
    out_channel, out_h, out_w = out_shape
    layers = []
    layers.append(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
    layers.append(nn.ELU(True))

    prev_channel_num = out_channel
    for idx in range(repeat_num):
      channel_num = out_channel * (idx + 1)
      layers.append(nn.Conv2d(prev_channel_num, channel_num, 3, 1, 1))
      layers.append(nn.ELU(True))
      if idx < repeat_num - 1:
        layers.append(nn.Conv2d(channel_num, channel_num, 3, 2, 1))
      else:
        layers.append(nn.Conv2d(channel_num, channel_num, 3, 1, 1))
      layers.append(nn.ELU(True))
      prev_channel_num = channel_num

    self.conv = nn.Sequential(*layers)

  def _internal_step(self, x):
    x = self.conv(x)
    _, c, h, w = x.size()
    dim = c * h * w
    self.fc = nn.Linear(dim, np.prod(self.out_shape)).to(dv)
    return x.view(-1, dim)

  def forward(self, x):
    x = self._internal_step(x)
    x = self.fc(x).view([-1] + self.out_shape)
    return x
    


class Decoder(nn.Module):
  def __init__(self, in_shape, out_shape, repeat_num):
    super(Decoder, self).__init__()
    in_channel, in_h, in_w = in_shape
    out_channel, out_h, out_w = out_shape
    layers = []
    for idx in range(repeat_num):
      layers.append(nn.Conv2d(in_channel, in_channel, 3, 1, 1))
      layers.append(nn.ELU(True))
      layers.append(nn.Conv2d(in_channel, in_channel, 3, 1, 1))
      layers.append(nn.ELU(True))
      if idx < repeat_num - 1:
        layers.append(nn.UpsamplingNearest2d(scale_factor=2))
    layers.append(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
    layers.append(nn.ELU(True))
    self.conv = nn.Sequential(*layers)
  
  def forward(self, x):
    x = self.conv(x)
    return x

class GeneratorCNN(nn.Module):
  def __init__(self, in_shape, out_shape, hidden_dim, repeat_num):
    super(GeneratorCNN, self).__init__()
    self.latent_shape = [hidden_dim, 8, 8]
    assert len(in_shape) == 2, "Input noise has shape, bs * noise_dim!"
    self.fc = nn.Linear(in_shape[-1], np.prod(self.latent_shape))
    self.dec = Decoder(self.latent_shape, out_shape, repeat_num)

  def forward(self, x):
    x = self.fc(x).view([-1] + self.latent_shape)
    print("front decoder x.shape: ", x.size())
    x = self.dec(x)
    return x

class DiscriminatorCNN(nn.Module):
  def __init__(self, in_shape, out_shape, hidden_dim, repeat_num):
    super(DiscriminatorCNN, self).__init__()
    self.latent_shape = [hidden_dim, 8, 8]
    self.enc = Encoder(in_shape, self.latent_shape, repeat_num)
    self.dec = Decoder(self.latent_shape, out_shape, repeat_num)

  def forward(self, x):
    latent = self.enc(x)
    x = self.dec(latent)
    # return latent, x
    return x
    
    # Encoder
    # 怎么把conv1_output_dim推导出来，其实就是出来之后的空间结构
    # 要把它记住
