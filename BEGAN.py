# TODO:
# [x] Plot loss and meaningful params
# [ ] Modify network structure

import os
import torch
import torchvision
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import models.celeba as nets
from torch.utils.data import DataLoader

import utils

class BEGAN(utils.BaseModel):
  def __init__(self, config, dataset):
    super(BEGAN, self).__init__(config, dataset)
    self.batch_size = config.batch_size
    self.num_noise_dim = config.num_noise_dim

    self.lr = config.lr
    self.gamma = config.gamma
    self.lambda_k = config.lambda_k

    self.build_model()

  def train(self):
    self.log = {}
    self.log['d_loss'] = []
    self.log['g_loss'] = []
    self.log['k_t'] = []
    self.log['lr'] = [self.config.lr]
    self.log['measure'] = [1]
    self.measure = {} # convergence measure
    self.measure['pre'] = []
    self.measure['pre'].append(1)
    self.measure['cur'] = []
    generated_images = []
    
    bs = self.batch_size
    dv = self.device
    AeLoss = nn.L1Loss().to(dv)

    z = torch.FloatTensor(bs, self.num_noise_dim).to(dv)
    z_fixed = torch.FloatTensor(bs, self.num_noise_dim).to(dv)
    z_fixed.normal_(0, 1)

    def _get_optimizer(lr):
      return optim.Adam(self.G.parameters(), lr=lr, betas=(self.config.beta1, self.config.beta2)), \
             optim.Adam(self.D.parameters(), lr=lr, betas=(self.config.beta1, self.config.beta2)),
    
    g_optim, d_optim = _get_optimizer(self.lr)
    dataloader = DataLoader(self.dataset, batch_size=bs, shuffle=True, num_workers=12)
    real_fixed_image = next(iter(dataloader))[0].to(dv)

    # Training...
    print('-'*25)
    print('Starting Training Loop...\n')
    print('Epochs: {}\nDataset: {}\nBatch size: {}\nLength of Dataloder: {}'
          .format(self.config.num_epoch, self.config.dataset, 
                  self.config.batch_size, len(dataloader)))
    print('-'*25)

    t0 = utils.ETimer() # train timer
    t1 = utils.ETimer() # epoch timer
    k_t = 0
    self.log['k_t'].append(k_t)

    self.D.train()
    for epoch in range(self.config.num_epoch):
      self.G.train()
      t1.reset()
      for num_iter, (image, _) in enumerate(dataloader):
        if image.size(0) != bs:
          break
        
        image = image.to(dv)
        z.normal_(0, 1)

        # Update discriminator.
        d_optim.zero_grad()
        _, d_real = self.D(image)
        d_loss_real = AeLoss(d_real, image)

        fake_image = self.G(z)
        _, d_fake = self.D(fake_image.detach())
        d_loss_fake = AeLoss(d_fake, fake_image.detach())

        d_loss = d_loss_real - k_t * d_loss_fake
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        d_loss.backward()
        d_optim.step()

        # Update generator.
        g_optim.zero_grad()
        _, d_fake = self.D(fake_image)
        g_loss = AeLoss(d_fake, fake_image)
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
        g_optim.step()

        # Convergence metric.
        balance = (self.gamma * d_loss_real - g_loss).item()
        temp_measure = d_loss_real + abs(balance)
        self.measure['cur'] = temp_measure.item()
        self.log['measure'].append(self.measure['cur'])
        # update k_t
        k_t += self.lambda_k * balance
        k_t = max(0, min(1, k_t))
        self.log['k_t'].append(k_t)

        # Print progress...
        if (num_iter+1) % 100 == 0:
          print('Epoch: ({:3.0f}/{:3.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, len(dataloader), 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
          img = self.generate(z_fixed, self.save_dir, epoch+1)
          self.autoencode(real_fixed_image, self.save_dir, epoch+1)
      # end of epoch
      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))

      if np.mean(self.measure['pre']) < np.mean(self.measure['cur']):
        self.lr *= 0.5
        g_optim, d_optim = _get_optimizer(self.lr)
      else:
        print('M_pre: ' + str(np.mean(self.measure['pre'])) + ', M_cur: ' + str(np.mean(self.measure['cur'])))
        self.measure['pre'] = self.measure['cur']
        self.measure['cur'] = []
      self.log['lr'].append(self.lr)
      
      if (epoch+1) % 1 == 0:
        img = self.generate(z_fixed, self.save_dir, epoch+1)
        generated_images.append(img)
        self.autoencode(real_fixed_image, self.save_dir, epoch+1, fake_image)

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    print('-'*50)
    utils.generate_animation(self.save_dir, generated_images)
    utils.plot_loss(self.log, self.save_dir)    
    self.plot_param(self.log, self.save_dir)

  
  def build_model(self):
    channel, height, width = self.dataset[0][0].size()
    assert height == width, "Height and width must equal."
    latent_dim = 0  # latent space vector dim
    if self.config.dataset == 'CelebA':
      repeat_num = int(np.log2(height)) - 2
      latent_dim = 64
    elif self.config.dataset == 'MNIST':
      repeat_num = int(np.log2(height)) - 1
      latent_dim = 32
    else:
      raise NotImplementedError
    noise_dim = self.num_noise_dim
    hidden_dim = self.config.hidden_dim
    self.G = nets.GeneratorCNN(noise_dim, channel, hidden_dim, repeat_num)
    self.D = nets.DiscriminatorCNN(channel, channel, hidden_dim, repeat_num)
    for i in [self.G, self.D]:
      i.apply(utils.weights_init)
      i.to(self.device)
      utils.print_network(i)
    return self.G, self.D
  
  def generate(self, noise, path, idx=None):
    self.G.eval()
    img_path = os.path.join(path, 'G-epoch-{}.png'.format(idx))
    from PIL import Image
    from torchvision.utils import make_grid
    tensor = self.G(noise)
    grid = make_grid(tensor, nrow=8, padding=2)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(img_path)
    return ndarr

  def autoencode(self, inputs, path, idx=None, fake_inputs=None):
    img_path = os.path.join(path, 'D-epoch-{}.png'.format(idx))
    _, img = self.D(inputs)
    vutils.save_image(img, img_path)
    if fake_inputs is not None:
      fake_img_path = os.path.join(path, 'D_fake-epoch-{}.png'.format(idx))
      _, fake_img = self.D(fake_inputs)
      vutils.save_image(fake_img, fake_img_path)
  
  def plot_param(self, log: dict, path: str):
    plt.style.use('ggplot')
    # plt.figure(figsize=(10, 5))
    plt.title('K_t')
    plt.plot(log['k_t'], linewidth=1)
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.savefig(path + '/k_t.png')
    plt.close('all')

    plt.title('Learning rate with epoch')
    plt.plot(log['lr'], linewidth=1)
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig(path + '/lr.png')
    plt.close('all')

    plt.title('Convergence measure')
    plt.plot(log['measure'], linewidth=1)
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.savefig(path + '/measure.png')
    plt.close('all')
