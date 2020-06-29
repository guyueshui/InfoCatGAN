# Networks for SVHN dataset. torch.Size([bs, 3, 32, 32])
from . import *

#======= following arch from InfoGAN paper ========
class Generator(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Generator, self).__init__()

    self.fc = nn.Sequential(
      nn.Linear(in_dim, 448*2*2),
      nn.BatchNorm1d(448*2*2),
      nn.ReLU(),
    )

    # torch.Size([bs, 448, 2, 2])
    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(448, 256, 4, 2, 1),  # bs x 256 x 4 x 4
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.ConvTranspose2d(256, 128, 4, 2, 1),  # bs x 128 x 8 x 8
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, 4, 2, 1),   # bs x 64 x 16 x 16
      nn.ReLU(),
      nn.ConvTranspose2d(64, out_dim, 4, 2, 1),     # bs x 3 x 32 x 32
      nn.Tanh(),
    )

  def forward(self, x):
    x = self.fc(x).view(-1, 448, 2, 2)
    x = self.deconv(x)
    return x


class Dbody(nn.Module):
  def __init__(self, in_dim, out_dim=256):
    super(Dbody, self).__init__()

    # torch.Size([bs, 3, 32, 32])
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 64, 4, 2, 1), # 16 x 16
      nn.LeakyReLU(0.1, True),
      nn.Conv2d(64, 128, 4, 2, 1),    # 8 x 8
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, True),
      nn.Conv2d(128, out_dim, 4, 2, 1),   # 4 x 4
      nn.BatchNorm2d(out_dim),
      nn.LeakyReLU(0.1, True),
    )
    # torch.Size([bs, 256, 4, 4])

  def forward(self, x):
    x = self.conv(x)
    return x

class Dhead(nn.Module):
  def __init__(self, in_dim, out_dim=1):
    super(Dhead, self).__init__()

    self.fc = nn.Sequential(
      ReshapeLayer((-1, in_dim*4*4)),
      nn.Linear(in_dim*4*4, out_dim),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.fc(x)
    return x

class Qhead(nn.Module):
  def __init__(self, in_dim, num_disc_code, num_cont_code):
    super(Qhead, self).__init__()
    self.num_disc_code = num_disc_code
    
    self.main = nn.Sequential(
      ReshapeLayer((-1, in_dim*4*4)),
      nn.Linear(in_dim*4*4, 128),
      nn.BatchNorm1d(128),
      nn.LeakyReLU(0.1, True),
      #nn.Linear(128, 10*num_disc_code + num_cont_code)
    )
    self.dis = nn.Linear(128, 10*num_disc_code)
    self.mu = nn.Linear(128, num_cont_code)
    self.var = nn.Linear(128, num_cont_code)
    self.softmax = nn.Softmax(dim=1)  # Each row sums to 1.
  
  def forward(self, x):
    x = self.main(x)
    disc_logits = self.dis(x)
    mu = self.mu(x)
    var = self.var(x).exp()
    return disc_logits, mu, var

#========= following arch is from ==============
# https://github.com/Natsu6767/InfoGAN-PyTorch/blob/master/models/svhn_model.py
class Natsu_Generator(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Natsu_Generator, self).__init__()
    self.in_dim = in_dim
    self.deconv = nn.Sequential(
      ReshapeLayer((-1, in_dim, 1, 1)),
      nn.ConvTranspose2d(in_dim, 448, 2, 1, bias=False),
      nn.BatchNorm2d(448),
      nn.ReLU(),

      nn.ConvTranspose2d(448, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(),

      nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
      nn.Tanh(),
    )

  def forward(self, x):
    x = self.deconv(x)
    return x

class Natsu_DBody(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Natsu_DBody, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(in_dim, 64, 4,2,1, bias=False),
      nn.LeakyReLU(0.1, True),
      nn.Conv2d(64, 128, 4,2,1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, True),
      nn.Conv2d(128, 256, 4,2,1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.1, True),
    )
  
  def forward(self, x):
    x = self.main(x)
    return x

class Natsu_DHead(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Natsu_DHead, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 1, 4),
      nn.Sigmoid(),
    )
  def forward(self, x):
    x = self.conv(x)
    return x

class Natsu_QHead(nn.Module):
  def __init__(self, in_dim, disc_dim, conc_dim):
    super(Natsu_QHead, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_dim, 128, 4, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, True)
    )

    self.conv_disc = nn.Conv2d(128, 10*disc_dim, 1)
    self.conv_mu = nn.Conv2d(128, conc_dim, 1)
    self.conv_var = nn.Conv2d(128, conc_dim, 1)

  def forward(self, x):
    x = self.conv(x)
    disc_logits = self.conv_disc(x).squeeze()
    mu = self.conv_mu(x).squeeze()
    var = self.conv_var(x).squeeze().exp()
    return disc_logits, mu, var
  

class Natsu_CATD(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(Natsu_CATD, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(in_dim, 64, 4,2,1, bias=False),
      nn.LeakyReLU(0.1, True),
      nn.Conv2d(64, 128, 4,2,1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, True),
      nn.Conv2d(128, 256, 4,2,1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.1, True),

      nn.Conv2d(256, 128, 4, bias=False),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, out_dim, 1),
      ReshapeLayer((-1, out_dim)),
    )

  def forward(self, x):
    logits = self.main(x)
    return logits

#=========== following arch from ===================
# https://github.com/xinario/catgan_pytorch/blob/master/catgan_cifar10.py

nlayers = 64

class XCATG(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(XCATG, self).__init__()
    self.main = nn.Sequential(
      ReshapeLayer((-1, in_dim, 1, 1)),
      nn.ConvTranspose2d(in_dim, nlayers*4, 4,1,0, bias=False), # 4x4
      nn.BatchNorm2d(nlayers*4),
      nn.LeakyReLU(0.2, inplace=True),

      nn.ConvTranspose2d(nlayers*4, nlayers*2, 4,2,1, bias=False),  # 8x8
      nn.BatchNorm2d(nlayers*2),
      nn.LeakyReLU(0.2, True),

      nn.ConvTranspose2d(nlayers*2, nlayers, 4,2,1, bias=False),  # 16x16
      nn.BatchNorm2d(nlayers),
      nn.LeakyReLU(0.2, True),

      nn.ConvTranspose2d(nlayers, out_dim, 4,2,1, bias=False),  # 32x32
      nn.Tanh()
    )
  
  def forward(self, x):
    x = self.main(x)
    return x
  

class XCATD(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(XCATD, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(in_dim, nlayers, 4,2,1, bias=False),  # 16x16
      nn.BatchNorm2d(nlayers),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),

      nn.Conv2d(nlayers, nlayers*2, 4,2,1, bias=False), # 8x8
      nn.BatchNorm2d(nlayers*2),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),

      nn.Conv2d(nlayers*2, nlayers*4, 4,2,1, bias=False), # 4x4
      nn.BatchNorm2d(nlayers*4),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),
      
      nn.Conv2d(nlayers*4, nlayers*4, 4), # 1x1
      nn.BatchNorm2d(nlayers*4),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),

      nn.Conv2d(nlayers*4, out_dim, 1),
      ReshapeLayer((-1, out_dim)),
    )

  def forward(self, x):
    x = self.main(x)
    return x

class XDbody(nn.Module):
  def __init__(self, in_dim, out_dim=nlayers*4):
    super(XDbody, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(in_dim, nlayers, 4,2,1, bias=False),  # 16x16
      nn.BatchNorm2d(nlayers),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),

      nn.Conv2d(nlayers, nlayers*2, 4,2,1, bias=False), # 8x8
      nn.BatchNorm2d(nlayers*2),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),

      nn.Conv2d(nlayers*2, nlayers*4, 4,2,1, bias=False), # 4x4
      nn.BatchNorm2d(nlayers*4),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),
      
      nn.Conv2d(nlayers*4, nlayers*4, 4), # 1x1
      nn.BatchNorm2d(nlayers*4),
      nn.LeakyReLU(0.2, True),
      nn.Dropout(0.5),

      #nn.Conv2d(nlayers*4, out_dim, 1),
      #ReshapeLayer((-1, out_dim)),
    )
  
  def forward(self, x):
    x = self.main(x)
    return x

class XDhead(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(XDhead, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(in_dim, out_dim, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.main(x).squeeze()
    return x

class XQhead(nn.Module):
  def __init__(self, in_dim, dis_dim, con_dim):
    super(XQhead, self).__init__()
    self.disc = nn.Sequential(
      nn.Conv2d(in_dim, dis_dim, 1),
      ReshapeLayer((-1, dis_dim))
    )
    self.mu = nn.Conv2d(in_dim, con_dim, 1)
    self.var = nn.Conv2d(in_dim, con_dim, 1)
  
  def forward(self, x):
    logits = self.disc(x)
    mu = self.mu(x)
    var = self.var(x).exp()
    return logits, mu, var