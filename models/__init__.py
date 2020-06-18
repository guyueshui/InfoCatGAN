from __future__ import absolute_import
import torch
import torch.nn as nn

class ReshapeLayer(nn.Module):
    def __init__(self, out_shape):
        super(ReshapeLayer, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(self.out_shape)

class CovvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 act=None, bn=True, ps=0, dr=0.0):
        super(CovvLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if act is not None:
            layers.append(act)
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if ps > 1:
            layers.append( nn.MaxPool2d( kernel_size=(ps, ps) ) )
        if dr > 0:
            layers.append(nn.Dropout2d(dr))
        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main(x)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, sigma=0.1):
        super(GaussianNoiseLayer, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.sigma == 0:
            return x
        else:
            self.noise = torch.zeros_like(x)
            self.noise.normal_(0, self.sigma)
            return x + self.noise

class PerforatedUpsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(PerforatedUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(2, stride=2)
        
    
    def forward(self, x):
        upx = self.upsample(x)
        _, idx = self.maxpool(upx)
        x = self.maxunpool(x, idx)
        return x
