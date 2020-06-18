import torch

def Dtest():
    a = torch.randn(23, 1, 28, 28)
    from models.mnist import D
    m = D(1, 10)
    out = m(a)
    print(out.size())
    print("pass")

def Gtest():
    a = torch.randn(23, 128)
    from models.mnist import G
    m = G(128, 1)
    out = m(a)
    print(out.size())
    print("pass")

if __name__ == "__main__":
    Gtest()