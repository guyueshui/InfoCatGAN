import torch

def Dtest():
    a = torch.randn(23, 3, 32, 32)
    from models.cifar10 import DCGAN_D
    m = DCGAN_D(3, 10)
    out = m(a)
    print(out.size())
    print("pass")

def Gtest():
    a = torch.randn(23, 128)
    from models.cifar10 import DCGAN_G
    m = DCGAN_G(128, 3)
    out = m(a)
    print(out.size())
    print("pass")

if __name__ == "__main__":
    import os
    fname = os.path.splitext(__file__)[0]
    print(fname)