from . import *
from torch.nn.utils import weight_norm

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        # torch.Size([bs, 1, 28, 28])
        layers = []
        layers += [
            ReshapeLayer((-1, 1, 28, 28)),
            CovvLayer(in_dim, 32, 5, 1, 0, act=nn.ReLU(), bn=True, ps=2, dr=0.5),   # 32, 12, 12
            CovvLayer(32, 64, 3, 1, 1, act=nn.ReLU(), bn=True, ps=1, dr=0),         # 64, 12, 12
            CovvLayer(64, 64, 3, 1, 1, act=nn.ReLU(), bn=True, ps=2, dr=0.5),       # 64, 6, 6
            CovvLayer(64, 128, 3, 1, 1, act=nn.ReLU(), bn=True, ps=1, dr=0),        # 128, 6, 6
            CovvLayer(128, 128, 3, 1, 0, act=nn.ReLU(), bn=True, ps=1, dr=0),       # 128, 4, 4
            nn.AvgPool2d(4)
        ]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.Softmax(dim=1),  # Each row sums to 1.
        )
    
    def forward(self, x):
        x = self.conv(x).squeeze()
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_dim, extra_dim=10, out_dim=28*28):
        super(Generator, self).__init__()
        self.blk1 = self._f_(in_dim + extra_dim, 500)
        self.blk2 = self._f_(500 + extra_dim, 500)
        self.blk3 = nn.Sequential(
            nn.Linear(500 + extra_dim, out_dim),
            nn.Sigmoid(),
            weight_norm(nn.BatchNorm1d(out_dim))
        )
    
    def _f_(self, in_dim, out_dim):
        seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Softplus(),
            nn.BatchNorm1d(out_dim)
        )
        return seq
    
    def forward(self, zac, y_g):
        assert zac.dim() == 2, "undesired input shape"
        if y_g.dim() < 2:
            y_g = nn.functional.one_hot(y_g.long(), num_classes=10)
            y_g = y_g.float()
        x = torch.cat([zac, y_g], dim=1)
        x = torch.cat([self.blk1(x), y_g], dim=1)
        x = torch.cat([self.blk2(x), y_g], dim=1)
        x = self.blk3(x).view(-1, 1, 28, 28)
        return x
        
class Discriminator(nn.Module):
    def __init__(self, in_dim=28*28, extra_dim=10, out_dim=1):
        super(Discriminator, self).__init__()
        self.nl_data = GaussianNoiseLayer(0.3)
        self.nl = GaussianNoiseLayer(0.5)
        self.reshape = ReshapeLayer((-1, in_dim))
        self.fc1 = self._f_(in_dim + extra_dim, 1000)
        self.fc2 = self._f_(1000 + 2*extra_dim, 500)
        self.fc3 = self._f_(500 + extra_dim, 250)
        self.fc4 = self._f_(250 + extra_dim, 250)
        self.fc5 = self._f_(250 + extra_dim, 250)
        self.fc6 = nn.Sequential(
            weight_norm(nn.Linear(250 + extra_dim, 1)),
            nn.Sigmoid(),
        )

    def _f_(self, in_dim, out_dim):
        seq = nn.Sequential(
            weight_norm( nn.Linear(in_dim, out_dim) ),
            nn.ReLU()
        )
        return seq
    
    def forward(self, x, y):
        if y.dim() < 2:
            y = nn.functional.one_hot(y.long(), num_classes=10)
            y = y.float()
        x = self.reshape(x)
        x = self.nl_data(x)
        x = torch.cat([x, y], dim=1)

        x = torch.cat([self.fc1(x), y], dim=1)
        x = self.nl(x)
        x = torch.cat([x, y], dim=1)

        x = self.fc2(x)
        x = self.nl(x)
        x = torch.cat([x, y], dim=1)

        x = self.fc3(x)
        x = self.nl(x)
        x = torch.cat([x, y], dim=1)

        x = self.fc4(x)
        x = self.nl(x)
        x = torch.cat([x, y], dim=1)

        x = self.fc5(x)
        x = self.nl(x)
        x = torch.cat([x, y], dim=1)

        x = self.fc6(x)
        return x


#=========== following arch from CatGAN paper ===========
"""
- bn
- dropout
- perforated upsampling
- add noise to hidden layer
- leaky rate is 0.1
"""
class D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(D, self).__init__()
        self.out_dim = out_dim
        self.main = nn.Sequential(
            GaussianNoiseLayer(0.3),
            nn.Conv2d(in_dim, 32, 5, 1, 1), # 28->26
            nn.LeakyReLU(0.1, inplace=True), # before bn or after bn?
            nn.BatchNorm2d(32),
            GaussianNoiseLayer(0.3),

            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),  # 26->12
            nn.BatchNorm2d(32),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(32, 64, 3, 1, 1), # 12->12
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(64),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(64, 64, 3, 1, 1), # 12->12
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(64),
            GaussianNoiseLayer(0.3),

            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)), # 12->5
            nn.BatchNorm2d(64),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(64, 128, 3, 1, 1),    # 5->5
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(128),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(128, out_dim, 1, 1, 0), # 5->5
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(out_dim),
            GaussianNoiseLayer(0.3),

            nn.AvgPool2d(5),
            nn.BatchNorm2d(out_dim),
            GaussianNoiseLayer(0.1),
        )

        self.softmax = nn.Softmax(dim=1) # Each row sums to 1.

    
    def forward(self, x):
        x = self.main(x).view(-1, self.out_dim)
        x = self.softmax(x)
        return x
        

class G(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(G, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 8*8*96),
            nn.BatchNorm1d(8*8*96),
            GaussianNoiseLayer(0.3),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 96 x 8 x 8
        self.conv = nn.Sequential(
            PerforatedUpsample(),  # 8->16
            nn.Conv2d(96, 96, 3, 1, 1),

            nn.Conv2d(96, 64, 5, 1, 1), # 16->14
            nn.BatchNorm2d(64),
            GaussianNoiseLayer(0.3),
            nn.LeakyReLU(0.1, True),

            PerforatedUpsample(), # 14->28
            nn.Conv2d(64, 64, 3, 1, 1),

            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            GaussianNoiseLayer(0.3),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, out_dim, 5, 1, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x).view(-1, 96, 8, 8)
        x = self.conv(x)
        return x


#====== following arch from InfoGAN paper =======
class InfoGAN_G(nn.Module):
  'Arch from InfoGAN paper.'
  def __init__(self, in_dim=74, out_dim=1):
      super(InfoGAN_G, self).__init__()

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

class InfoGAN_D(nn.Module):
    def __init__(self, in_dim, out_dim, nclass, nconcode):
        super(InfoGAN_D, self).__init__()
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
            nn.LeakyReLU(0.2)
        )

        self.d = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        self.q = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        self.qdis = nn.Linear(128, nclass)
        self.qmu = nn.Linear(128, nconcode)
        self.qvar = nn.Linear(128, nconcode)
    
    def forward(self, x):
        x = self.conv(x).view(-1, 128*7*7)
        x = self.fc(x)
        prob = self.d(x)
        qstuff = self.q(x)
        dis_logits = self.qdis(qstuff)
        qmu = self.qmu(qstuff)
        qvar = self.qvar(qstuff).exp()
        return prob, dis_logits, (qmu, qvar)