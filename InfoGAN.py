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

class InfoGAN(utils.BaseModel):
  def __init__(self, config, dataset):
    super(InfoGAN, self).__init__(config, dataset)
    self.num_disc_code = 1
    self.num_cont_code = 2
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
    real_label = 1
    fake_label = 0

    z = torch.FloatTensor(bs, self.z_dim).to(dv)
    disc_c = torch.FloatTensor(bs, self.cat_dim * self.num_disc_code).to(dv)
    cont_c = torch.FloatTensor(bs, self.num_cont_code).to(dv)
    # Fixed random variables.
    fixz, cdis, c1, c2, c0 = self.generate_fix_noise()
    noise_fixed = np.hstack([fixz, cdis, c0])
    fixed_noise_1 = np.hstack([fixz, cdis, c1])
    fixed_noise_2 = np.hstack([fixz, cdis, c2])
    # NOTE: dtype should exactly match the network weight's type!
    noise_fixed = torch.as_tensor(noise_fixed, dtype=torch.float32).to(dv)
    fixed_noise_1 = torch.as_tensor(fixed_noise_1, dtype=torch.float32).to(dv)
    fixed_noise_2 = torch.as_tensor(fixed_noise_2, dtype=torch.float32).to(dv)

    DLoss = nn.BCELoss().to(dv)
    QdiscLoss = nn.CrossEntropyLoss().to(dv)
    # QcontLoss = nn.MSELoss().to(dv)
    QcontLoss = utils.LogGaussian()

    g_step_params = [{'params': self.G.parameters()}, {'params': self.Q.parameters()}]
    d_step_params = [{'params': self.FD.parameters()}, {'params': self.D.parameters()}]
    g_optim = optim.Adam(g_step_params, lr=1e-3, betas=(0.5, 0.99))
    d_optim = optim.Adam(d_step_params, lr=2e-4, betas=(0.5, 0.99))
      
    dataloader = DataLoader(self.dataset, batch_size=bs, shuffle=True, num_workers=4)
    tot_iters = len(dataloader)

    # Training...
    print('-'*25)
    print('Starting Training Loop...\n')
    print('Epochs: {}\nDataset: {}\nBatch size: {}\nLength of Dataloder: {}'
          .format(self.config.num_epoch, self.config.dataset, 
                  self.config.batch_size, len(dataloader)))
    print('-'*25)

    t0 = utils.ETimer() # train timer
    t1 = utils.ETimer() # epoch timer

    self.FD.train()
    self.D.train()
    for epoch in range(self.config.num_epoch):
      self.G.train()
      self.Q.train()
      t1.reset()
      for num_iter, (image, _) in enumerate(dataloader):
        if image.size(0) != bs:
          break
        
        image = image.to(dv)

        # Guided by https://github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        # This may be helpful for generator convergence.
        if self.config.instance_noise:
          instance_noise = torch.zeros_like(image)
          std = -0.1 / tot_iters * num_iter + 0.1
          instance_noise.normal_(0, std)
          image = image + instance_noise

        # Update discriminator.
        d_optim.zero_grad()
        d_real = self.D(self.FD(image))
        labels = torch.full_like(d_real, real_label, device=dv)
        d_loss_real = DLoss(d_real, labels)
        d_loss_real.backward()

        noise, idx = self.generate_noise(z, disc_c, cont_c)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          fake_image = fake_image + instance_noise

        d_fake = self.D(self.FD(fake_image.detach()))
        labels.fill_(fake_label)
        d_loss_fake = DLoss(d_fake, labels) 
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        d_optim.step()

        # Update generator.
        g_optim.zero_grad()
        d_body_out = self.FD(fake_image)
        d_fake = self.D(d_body_out)
        labels.fill_(real_label)
        reconstruct_loss = DLoss(d_fake, labels)

        q_logits, mu, var = self.Q(d_body_out)
        targets = torch.LongTensor(idx).to(dv)
        q_loss_disc = QdiscLoss(q_logits, targets) * 0.8
        q_loss_conc = QcontLoss(cont_c, mu, var) * 0.2

        g_loss = reconstruct_loss + q_loss_disc + q_loss_conc
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
        img = self.generate(self.G, fixed_noise_1, 'G-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)
      if (epoch+1) % 25 == 0:
        self.save_model(self.save_dir, epoch+1, *self.models)

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    self.save_model(self.save_dir, self.config.num_epoch, *self.models)
    print('-'*50)

    # Manipulating continuous latent codes.
    self.generate(self.G, fixed_noise_1, 'c1-final.png')
    self.generate(self.G, fixed_noise_2, 'c2-final.png')

    utils.generate_animation(self.save_dir, generated_images)
    utils.plot_loss(self.log, self.save_dir)    
  
  def build_model(self):
    channel, height, width = self.dataset[0][0].size()
    assert height == width, "Height and width must equal."
    # repeat_num = int(np.log2(height)) - 1
    # hidden_dim = self.config.hidden_dim
    noise_dim = self.z_dim + self.cat_dim * self.num_disc_code + self.num_cont_code
    latent_dim = 1024 # embedding latent vector dim
    import models.mnist as nets
    self.G = nets.OfficialGenerator(noise_dim, channel)
    self.FD = nets.OfficialDbody(channel, latent_dim)
    self.D = nets.OfficialDhead(latent_dim, 1)
    self.Q = nets.OfficialQ(latent_dim, self.cat_dim, self.num_cont_code)
    networks = [self.G, self.FD, self.D, self.Q]
    for i in networks:
      i.apply(utils.weights_init)
      i.to(self.device)
    return networks

  def generate_noise(self, z, disc_c, cont_c, prob=None):
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
    cont_c.uniform_(-1, 1)
    noise = torch.cat([z, disc_c, cont_c], dim=1).view(batch_size, -1)
    return noise, idx

  def generate_fix_noise(self):
    'Generate fix noise for image generating during the whole training.'
    fixz = torch.Tensor(100, self.z_dim).normal_(0, 1.0)

    c = np.linspace(-1, 1, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])
    c3 = np.random.uniform(-1, 1, (100, 2))

    idx = np.arange(self.cat_dim).repeat(10)
    one_hot = np.zeros((100, self.cat_dim))
    one_hot[range(100), idx] = 1
    
    return fixz.numpy(), one_hot, c1, c2, c3

  #def generate(self, noise, fname):
  #  'Generate fake images using generator.'
  #  self.G.eval()
  #  img_path = os.path.join(self.save_dir, fname)
  #  from PIL import Image
  #  from torchvision.utils import make_grid
  #  tensor = self.G(noise)
  #  grid = make_grid(tensor, nrow=10, padding=2)
  #  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  #  ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
  #  im = Image.fromarray(ndarr)
  #  im.save(img_path)
  #  return ndarr

  def raw_classify(self, imgs):
    imgs = imgs.to(self.device)
    self.FD.eval()
    self.Q.eval()
    with torch.no_grad():
      logits, _, _ = self.Q(self.FD(imgs))
    return logits.cpu().numpy()
