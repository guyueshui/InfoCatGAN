# Networks for MNIST dataset. torch.Size([batch_size, 1, 28, 28])

import torch.nn as nn
import numpy as np

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

class Encoder(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
    super(Encoder, self).__init__()
    self.latent_shape = [hidden_dim, 7, 7]
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
    self.flat_dim = repeat_num*hidden_dim*7*7
    self.fc = nn.Linear(self.flat_dim, out_dim)

  def forward(self, x):
    x = self.conv(x).view(-1, self.flat_dim)
    # print("encoder.out.size:", x.size())
    x = self.fc(x)
    return x
    

class Decoder(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
    super(Decoder, self).__init__()

    self.latent_shape = [hidden_dim, 7, 7]
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

# class GeneratorCNN(nn.Module):
#   def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
#     super(GeneratorCNN, self).__init__()
#     self.latent_dim = 32
#     self.fc = nn.Linear(in_dim, self.latent_dim)
#     self.dec = Decoder(self.latent_dim, out_dim, hidden_dim, repeat_num)
#   
#   def forward(self, x):
#     x = self.fc(x)
#     x = self.dec(x)
#     return x

class GeneratorCNN(nn.Module):
  def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
    super(GeneratorCNN, self).__init__()

    self.fc = nn.Sequential(
      nn.Linear(in_dim, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Linear(1024, 128 * 7 * 7), # bs x 128*7*7
      nn.BatchNorm1d(128 * 7 * 7),
      nn.ReLU(),
    )

    # torch.Size([bs, 128, 7, 7])
    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(128, 64, 4, 2, 1), # bs x 64 x 14 x 14
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, out_dim, 4, 2, 1), # bs x output_dim x 28 x 28
      nn.Tanh(),
    )
    # torch.Size([bs, 1, 28, 28])
  
  def forward(self, x):
    x = self.fc(x).view(-1, 128, 7, 7)
    x = self.deconv(x)
    return x


class DiscriminatorCNN(nn.Module):
  # Architecture from: https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/BEGAN.py
  def __init__(self, in_dim, out_dim, hidden_dim, repeat_num):
    super(DiscriminatorCNN, self).__init__()
    self.down_shape = [128, 7, 7]
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 64, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(64, 128, 4, 2, 1),
      nn.LeakyReLU(0.2),
    )
    self.fc1 = nn.Sequential(
      nn.Linear(np.prod(self.down_shape), 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2),
    )
    self.fc2 = nn.Sequential(
      nn.Linear(1024, np.prod(self.down_shape)),
      nn.BatchNorm1d(np.prod(self.down_shape)),
      nn.LeakyReLU(0.2),
    )
    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(128, 64, 4, 2, 1),
      nn.LeakyReLU(0.2),
      nn.ConvTranspose2d(64, out_dim, 4, 2, 1),
      nn.Tanh(),
    )

  def forward(self, x):
    x = self.conv(x).view(-1, np.prod(self.down_shape))
    latent = self.fc1(x)  # bs x 32
    x = self.fc2(latent).view([-1] + self.down_shape)
    x = self.deconv(x)
    return latent, x


class Qhead(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Qhead, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(1024, 128),
      nn.BatchNorm1d(128),
      nn.LeakyReLU(0.2),
      nn.Linear(128, out_dim),
      nn.Tanh(),
    )
  
  def forward(self, x):
    x = self.fc(x)
    return x 

class DProb(nn.Module):
  def __init__(self, in_dim=1024, out_dim=1):
    super(DProb, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.Sigmoid(),
    )
  
  def forward(self, x):
    x = self.fc(x)
    return x 

#================ below are not began structure ===========

# class GeneratorCNN(nn.Module):
#   def __init__(self, in_dim, out_dim, hidden_dim):
#     super(GeneratorCNN, self).__init__()
#     self.latent_shape = [hidden_dim, 7, 7]
#     flat_dim = np.prod(self.latent_shape)
#     self.fc = nn.Sequential(
#       nn.Linear(in_dim, 1024),
#       nn.BatchNorm1d(1024),
#       nn.ReLU(),
#       nn.Linear(1024, flat_dim),
#       nn.BatchNorm1d(flat_dim),
#       nn.ReLU(),
#     )
#     self.deconv = nn.Sequential(
#       nn.ConvTranspose2d(hidden_dim, 64, 4, 2, 1),
#       nn.BatchNorm2d(64),
#       nn.ReLU(),
#       nn.ConvTranspose2d(64, out_dim, 4, 2, 1),
#       nn.Tanh(),
#     )
#   
#   def forward(self, x):
#     x = self.fc(x).view([-1] + self.latent_shape)
#     x = self.deconv(x)
#     return x
# 
# class Dbody(nn.Module):
#   def __init__(self, in_dim, out_dim):
#     super(Dbody, self).__init__()
#     # Encoder
#     self.conv = nn.Sequential(
#       nn.Conv2d(in_dim, 64, 4, 2, 1),
#       nn.LeakyReLU(0.2),
#       nn.Conv2d(64, 128, 4, 2, 1),
#       nn.BatchNorm2d(128),
#       nn.LeakyReLU(0.2),
#     )
# 
#     self.fc = nn.Sequential(
#       nn.Linear(128*7*7, out_dim),
#       nn.BatchNorm1d(out_dim),
#       nn.LeakyReLU(0.2),
#     )
#     
#   def forward(self, x):
#     x = self.conv(x).view(-1, 128*7*7)
#     x = self.fc(x)
#     return x
# 
# class Dhead(nn.Module):
#   def __init__(self, in_dim, out_dim):
#     super(Dhead, self).__init__()
#     self.main = nn.Sequential(
#       nn.Linear(in_dim, 1),
#       nn.Sigmoid(),
#     )
#     
#   def forward(self, x):
#     x = self.main(x)
#     return x
# 
# class Qhead(nn.Module):
#   def __init__(self, in_dim, cat_dim=10, num_cont_code=2):
#     super(Qhead, self).__init__()
# 
#     self.fc = nn.Sequential(
#       nn.Linear(in_dim, 128),
#       nn.BatchNorm1d(128),
#       nn.LeakyReLU(0.2),
#       # nn.Linear(128, self.dis_dim + self.con_dim)
#     )
# 
#     self.disc = nn.Linear(128, cat_dim)
#     self.mu = nn.Sequential(
#       nn.Linear(128, num_cont_code),
#     #  nn.Tanh(),
#     )
#     self.var = nn.Linear(128, num_cont_code)
# 
#   def forward(self, x):
#     x = self.fc(x)
#     disc_logits = self.disc(x)
#     mu = self.mu(x)
#     var = self.var(x).exp() # use exp to ensure positivity
#     return disc_logits, mu, var
