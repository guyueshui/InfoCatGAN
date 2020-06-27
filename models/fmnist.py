from . import *
from torch.nn.utils import weight_norm

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
            #GaussianNoiseLayer(0.1),
        )

        self.softmax = nn.Softmax(dim=1) # Each row sums to 1.

    
    def forward(self, x):
        x = self.main(x).view(-1, self.out_dim)
        # NOTE: nn.CrossEntropyLoss needs logits instead of probs.
        #x = self.softmax(x)
        return x
        

class SEP_D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SEP_D, self).__init__()
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
        )

        self.dhead = nn.Sequential(
            nn.Conv2d(128, out_dim, 1, 1, 0), # 5->5
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(out_dim),
            GaussianNoiseLayer(0.3),

            nn.AvgPool2d(5),
            nn.BatchNorm2d(out_dim),
            #GaussianNoiseLayer(0.1),
        )

        self.softmax = nn.Softmax(dim=1) # Each row sums to 1.

    
    def forward(self, x):
        feat = self.main(x)
        real_logits = self.dhead(feat).view(-1, self.out_dim)
        # NOTE: nn.CrossEntropyLoss needs logits instead of probs.
        #x = self.softmax(x)
        return real_logits, feat

class G(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(G, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 8*8*96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(8*8*96),
            GaussianNoiseLayer(0.3),
        )

        # 96 x 8 x 8
        self.conv = nn.Sequential(
            PerforatedUpsample(),  # 8->16
            nn.Conv2d(96, 96, 3, 1, 1),

            nn.Conv2d(96, 64, 5, 1, 1), # 16->14
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(64),
            GaussianNoiseLayer(0.3),

            PerforatedUpsample(), # 14->28
            nn.Conv2d(64, 64, 3, 1, 1),

            nn.Conv2d(64, 64, 5, 1, 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(64),
            GaussianNoiseLayer(0.3),

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