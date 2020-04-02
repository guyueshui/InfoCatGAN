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
      nn.ConvTranspose2d(64, out_dim, kernel_size=4, stride=2, padding=1, bias=False),
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


class OfficialGenerator(nn.Module):
  'G = FrontG + Generator'
  def __init__(self, in_dim=74, out_dim=1):
    super(Generator, self).__init__()

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

#============= following arch from ===============
# https://raw.githubusercontent.com/minlee077/CATGAN-pytorch/master/notebooks/CATGAN.ipynb
def conv_bn_lrelu_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.Sequential(
    nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
    nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    nn.LeakyReLU(0.2)
  )
def conv_bn_lrelu_drop_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.Sequential(
    nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
    nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.5)
  )
def tconv_bn_relu_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.Sequential(
    nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
    nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    nn.ReLU()
  )

def tconv_bn_lrelu_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.Sequential(
    nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
    nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
    nn.ReLU()
  )
  
def tconv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

def conv_lrelu_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.Sequential(
    nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
    nn.LeakyReLU(0.2)
  )

def conv_lrelu_drop_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):
  return nn.Sequential(
    nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
    nn.LeakyReLU(0.2)
  )

def fc_bn_layer(in_features,out_features):
  return nn.Sequential(
    nn.Linear(in_features,out_features),
    nn.BatchNorm1d(out_features)
  )

def fc_layer(in_features,out_features):
  return nn.Linear(in_features,out_features)

import math
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

s_h, s_w = 28, 28
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
gf_dim = 32
df_dim = 32

class SomeG(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(SomeG,self).__init__()
    self.fc_layer1 = fc_layer(in_dim,s_h8*s_w8*gf_dim*4)
    self.bn_layer1 = nn.BatchNorm2d(gf_dim*4)#4x4
    self.up_sample_layer2 = tconv_bn_relu_layer(gf_dim*4,gf_dim*2,3,stride=2,padding=1)#7x7
    self.up_sample_layer3 = tconv_bn_relu_layer(gf_dim*2,gf_dim,4,stride=2,padding=1)#14x14
    self.up_sample_layer4 = tconv_layer(gf_dim,out_dim,4,stride=2,padding=1)#28x28
    self.tanh = nn.Tanh()


  def forward(self, x):
    x = self.fc_layer1(x)
    x = x.view(-1,gf_dim*4,s_h8,s_w8)
    x = self.bn_layer1(x)
    x = self.up_sample_layer2(x)
    x = self.up_sample_layer3(x)
    x = self.up_sample_layer4(x)
    x = self.tanh(x)
    return x

import torch
class SomeD(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(SomeD,self).__init__()
    self.down_sample_layer1 = conv_lrelu_layer(in_dim,df_dim,4,stride=2,padding=1)# 14x14
    self.down_sample_layer2 = conv_bn_lrelu_layer(df_dim,df_dim*2,4,stride=2,padding=1)#7x7
    self.down_sample_layer3 = conv_bn_lrelu_layer(df_dim*2,df_dim*4,3,stride=2,padding=1)
    self.fc_layer4 = fc_layer(df_dim*4*s_h8*s_w8,out_dim)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    x = self.down_sample_layer1(x)
    x = self.down_sample_layer2(x)
    x = self.down_sample_layer3(x)
    x = torch.flatten(x,1)
    x = self.fc_layer4(x)
    simplex = self.softmax(x)
    return simplex, x
