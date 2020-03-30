import os
import torch
import torchvision
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import utils

class InfoCatGAN(utils.BaseModel):
  def __init__(self, config, dataset):
    super(InfoCatGAN, self).__init__(config, dataset)
    self.num_disc_code = 1
    self.z_dim = config.num_noise_dim
    self.cat_dim = 10

    self.models = self.build_model()

  def train(self):
    self.log = {}
    self.log['d_loss'] = []
    self.log['g_loss'] = []
    generated_images = []
    
    bs = self.config.batch_size
    dv = self.device

    z = torch.FloatTensor(bs, self.z_dim).to(dv)
    disc_c = torch.FloatTensor(bs, self.cat_dim).to(dv)
    # Fixed random variables.
    fixz, cdis = self.generate_fix_noise()
    noise_fixed = np.hstack([fixz, cdis])
    noise_fixed = torch.as_tensor(noise_fixed, dtype=torch.float32).to(dv)

    celoss = nn.CrossEntropyLoss().to(dv)

    def _get_optimizer(lr):
      g_step_params = [{'params': self.G.parameters()}, {'params': self.D.parameters()}]
      d_step_params = [{'params': self.D.parameters()}]
      return optim.Adam(g_step_params, lr=0.0004, betas=(self.config.beta1, self.config.beta2)), \
             optim.Adam(d_step_params, lr=0.0002, betas=(self.config.beta1, self.config.beta2)),
      
    g_optim, d_optim = _get_optimizer(0)
    dataloader = DataLoader(self.dataset, batch_size=bs, shuffle=True, num_workers=4)

    # Training...
    print('-'*25)
    print('Starting Training Loop...\n')
    print('Epochs: {}\nDataset: {}\nBatch size: {}\nLength of Dataloder: {}'
          .format(self.config.num_epoch, self.config.dataset, 
                  self.config.batch_size, len(dataloader)))
    print('-'*25)

    t0 = utils.ETimer() # train timer
    t1 = utils.ETimer() # epoch timer

    self.D.train()
    for epoch in range(self.config.num_epoch):
      self.G.train()
      t1.reset()
      for num_iter, (image, _) in enumerate(dataloader):
        if image.size(0) != bs:
          break
        
        image = image.to(dv)

        # Update discriminator.
        d_optim.zero_grad()
        d_real_simplex, _ = self.D(image)
        # Minimize entropy to make certain prediction of real sample.
        ent_real = utils.Entropy(d_real_simplex)
        # Maximize marginal entropy over real samples to ensure equal usage.
        margin_ent_real = utils.MarginalEntropy(d_real_simplex)

        noise, idx = self.generate_noise(z, disc_c)
        fake_image = self.G(noise)
        d_fake_simplex, _ = self.D(fake_image.detach())
        # Maximize entropy to make uncertain prediction of fake sample.
        ent_fake = utils.Entropy(d_fake_simplex)

        d_loss = ent_real - margin_ent_real - ent_fake
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        d_loss.backward()
        d_optim.step()

        # Update generator.
        g_optim.zero_grad()
        d_fake_simplex, d_fake_logits = self.D(fake_image)
        # Fool D to make it believe the fake is real.
        ent_fake = utils.Entropy(d_fake_simplex)
        # Ensure equal usage of fake samples.
        margin_ent_fake = utils.MarginalEntropy(d_fake_simplex)
        targets = torch.LongTensor(idx).to(dv)
        binding_loss = celoss(d_fake_logits, targets) * 0.8

        g_loss = ent_fake - margin_ent_fake + binding_loss
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
        g_optim.step()

        # Print progress...
        if (num_iter+1) % 100 == 0:
          print('Epoch: ({:2.0f}/{:2.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, len(dataloader), 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
      # end of epoch
      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))

      if (epoch+1) % 1 == 0:
        img = self.generate(noise_fixed, 'G-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    self.save_model(self.save_dir, self.config.num_epoch, *self.models)
    print('-'*50)

    utils.generate_animation(self.save_dir, generated_images)
    utils.plot_loss(self.log, self.save_dir)    
  
  def build_model(self):
    import models.mnist as nets
    channel, height, width = self.dataset[0][0].size()
    assert height == width, "Height and width must equal."
    # repeat_num = int(np.log2(height)) - 1
    # hidden_dim = self.config.hidden_dim
    noise_dim = self.z_dim + self.cat_dim 
    self.G = nets.Generator(noise_dim, channel)
    self.D = nets.CatD(channel, self.cat_dim)
    networks = [self.G, self.D]
    for i in networks:
      i.apply(utils.weights_init)
      i.to(self.device)
    return networks

  def generate_noise(self, z, disc_c, prob=None):
    'Generate samples for G\'s input.'
    if prob is None:
      prob = np.array([1 / self.cat_dim]).repeat(self.cat_dim)
    else: 
      prob = prob.squeeze()
      assert len(prob) == self.cat_dim
    batch_size = z.size(0)
    onehot, idx = utils.Noiser.Category(prob.squeeze(), batch_size)
    z.normal_(0, 1)
    disc_c.copy_(torch.tensor(onehot))
    noise = torch.cat([z, disc_c], dim=1).view(batch_size, -1)
    return noise, idx

  def generate_fix_noise(self):
    'Generate fix noise for image generating during the whole training.'
    fixz = torch.Tensor(100, self.z_dim).normal_(0, 1.0)
    idx = np.arange(self.cat_dim).repeat(10)
    one_hot = np.zeros((100, self.cat_dim))
    one_hot[range(100), idx] = 1
    return fixz.numpy(), one_hot

  def generate(self, noise, fname):
    'Generate fake images using generator.'
    self.G.eval()
    img_path = os.path.join(self.save_dir, fname)
    from PIL import Image
    from torchvision.utils import make_grid
    tensor = self.G(noise)
    grid = make_grid(tensor, nrow=10, padding=2)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(img_path)
    return ndarr