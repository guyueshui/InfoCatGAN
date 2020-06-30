Code for paper
==============

"Classification Models Based on Generative Adversarial Networks with Mutual Information Regularization"

## Implemented models

- CatGAN
- InfoCatGAN
- InfoGAN classifier

## Environment

- pytorch 1.2.0
- torchvision 0.4.0
- python 3.6.7
- GTX 1080Ti *2

## Usage

Run for mnist:
```bash
# run catgan on mnist
python catgan_mnist.py --nlabeled <x> --seed <x> --tag <x> --gpu 0
# run infocatgan on mnist
python infocatgan_mnist.py --nlabeled <x> --seed <x> --tag <x>
# run catgan on fashion mnist
python catgan_fmnist.py
```

## Reference

- [infoGAN-pytorch](https://github.com/pianomania/infoGAN-pytorch)
- [Natsu6767/InfoGAN-PyTorch](https://github.com/Natsu6767/InfoGAN-PyTorch)
- [xinario/catgan_pytorch](https://github.com/xinario/catgan_pytorch)

