from . import *

#=============== the following arch from CatGAN paper ==============
class D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(D, self).__init__()
        self.out_dim = out_dim
        # torch.Size([bs, 3, 32, 32])
        self.main = nn.Sequential(
            GaussianNoiseLayer(0.3),
            nn.Conv2d(in_dim, 96, 3, 1, 1), # 32->32
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(96),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(96, 96, 3, 1, 1), # 32->32
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(96),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(96, 96, 3, 1, 1), # 32->32
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(96),
            GaussianNoiseLayer(0.3),

            nn.MaxPool2d(kernel_size=2, stride=2), # 32->16
            nn.BatchNorm2d(96),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(96, 192, 3, 1, 1), # 16->16
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(192),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(192, 192, 3, 1, 1), # 16->16
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(192),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(192, 192, 3, 1, 1), # 16->16
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(192),
            GaussianNoiseLayer(0.3),

            nn.MaxPool2d(kernel_size=2, stride=2), # 16->8
            nn.BatchNorm2d(192),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(192, 192, 3, 1, 1), # 8->8
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(192),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(192, 192, 1), # 8->8
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(192),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(192, out_dim, 1), # 8->8
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(out_dim),
            GaussianNoiseLayer(0.3),

            nn.AvgPool2d(8),
            nn.BatchNorm2d(out_dim),
            GaussianNoiseLayer(0.1)
        )

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.main(x).view(-1, self.out_dim)
        x = self.softmax(x)
        return x


class G(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(G, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 8*8*192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(8*8*192),
            GaussianNoiseLayer(0.3),
        )
        # torch.Size([bs, 192, 8, 8])
        self.conv = nn.Sequential(
            PerforatedUpsample(),   # 8->16
            nn.Conv2d(192, 192, 3, 1, 1),

            nn.Conv2d(192, 96, 5, 1, 2),  # 16->16
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(96),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(96, 96, 5, 1, 2),  # 16->16
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(96),
            GaussianNoiseLayer(0.3),

            PerforatedUpsample(),   # 16->32
            nn.Conv2d(96, 96, 3, 1, 1),

            nn.Conv2d(96, 96, 5, 1, 2),  # 16->16
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(96),
            GaussianNoiseLayer(0.3),

            nn.Conv2d(96, out_dim, 5, 1, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x).view(-1, 192, 8, 8)
        x = self.conv(x)
        return x


#============ following arch from =================
# https://github.com/AlexiaJM/Deep-learning-with-cats/blob/master/Generating%20cats/DCGAN.py

class DCGAN_G(nn.Module):
    def __init__(self, in_dim:int, out_shape:tuple):
        super(DCGAN_G, self).__init__()
        out_dim, h, w = out_shape
        assert h == w
        mult = h // 8
        G_h_size = 128

        layers = [ReshapeLayer((-1, in_dim, 1, 1))]
        layers += [
            nn.ConvTranspose2d(in_dim, mult*G_h_size, 4, bias=False),
            nn.BatchNorm2d(mult*G_h_size),
            nn.ReLU(),
        ]
        while mult > 1:
            layers += [
                nn.ConvTranspose2d(G_h_size*mult, G_h_size*(mult//2), 4,2,1, bias=False),
                nn.BatchNorm2d(G_h_size*(mult//2)),
            ]
            mult = mult // 2
        layers += [
            nn.ConvTranspose2d(G_h_size, out_dim, 4, 2, 1, bias=False),
            nn.Tanh()
        ]
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.main(x)
        return x


class DCGAN_D(nn.Module):
    def __init__(self, in_shape:tuple, out_dim:int):
        super(DCGAN_D, self).__init__()
        in_dim, h, w = in_shape
        self.out_dim = out_dim
        assert h == w
        D_h_size = 128

        layers = []
        layers += [
            nn.Conv2d(in_dim, D_h_size, 4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        image_size_new = h // 2
        mult = 1
        while image_size_new > 4:
            layers += [
                nn.Conv2d(D_h_size*mult, D_h_size*(2*mult), 4,2,1, bias=False),
                nn.BatchNorm2d(D_h_size*2*mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            image_size_new = image_size_new // 2
            mult *= 2
        layers += [
            nn.Conv2d(D_h_size*mult, out_dim, 4,1,0, bias=False),
            nn.Sigmoid()
        ]
        self.conv = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x