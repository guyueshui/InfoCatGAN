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
from fid import fid_score

class CatGAN(utils.BaseModel):
  def __init__(self, config, dataset):
    super(CatGAN, self).__init__(config, dataset)
    self.z_dim = config.num_noise_dim
    self.cat_dim = config.num_class
    self.models = self.build_model()

  def train(self):
    self.log = {}
    self.log['d_loss'] = []
    self.log['g_loss'] = []
    self.log['fid'] = []
    generated_images = []

    bs = self.config.batch_size
    dv = self.device
    one = torch.FloatTensor([1]).to(dv)
    mone = one * -1
    
    def _get_optimizer(lr):
      g_step_params = [{'params': self.G.parameters()}]
      d_step_params = [{'params': self.D.parameters()}]
      return optim.Adam(g_step_params, lr=1e-3, betas=(self.config.beta1, self.config.beta2)), \
             optim.Adam(d_step_params, lr=2e-4, betas=(self.config.beta1, self.config.beta2)),

    g_optim, d_optim = _get_optimizer(0)
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
    noise_fixed = torch.randn(100, self.z_dim).to(dv)

    self.D.train()
    for epoch in range(self.config.num_epoch):
      # Add FID score...
      if self.config.fid:
        imgs_cur_epoch = []

      t1.reset()
      self.G.eval()
      for num_iter, (image, _) in enumerate(dataloader):
        if image.size(0) != bs:
          break
        image = image.to(dv)

        # Update discriminator.
        d_optim.zero_grad()

        # Guided by https://github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        # This may be helpful for generator convergence.
        if self.config.instance_noise:
          instance_noise = torch.zeros_like(image)
          std = -0.1 / tot_iters * num_iter + 0.1
          instance_noise.normal_(0, std)
          image = image + instance_noise

        d_real_simplex = self.D(image)
        # Minimize entropy to make certain prediction of real sample.
        ent_real = utils.Entropy(d_real_simplex)
        ent_real.backward(one, retain_graph=True)
        # Maximize marginal entropy over real samples to ensure equal usage.
        margin_ent_real = utils.MarginalEntropy(d_real_simplex)
        margin_ent_real.backward(mone)

        noise = torch.randn(bs, self.z_dim).to(dv)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          # instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

        d_fake_simplex = self.D(fake_image.detach())
        # Maximize entropy to make uncertain prediction of fake sample.
        ent_fake = utils.Entropy(d_fake_simplex)
        ent_fake.backward(mone)

        d_loss = ent_real + margin_ent_real + ent_fake
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        d_optim.step()

        # Train generator.
        g_optim.zero_grad()

        noise = torch.randn(bs, self.z_dim).to(dv)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          # instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

        d_fake_simplex = self.D(fake_image)
        # Fool D to make it believe the fake is real.
        ent_fake = utils.Entropy(d_fake_simplex)
        ent_fake.backward(one, retain_graph=True)
        # Ensure equal usage of fake samples.
        margin_ent_fake = utils.MarginalEntropy(d_fake_simplex)
        margin_ent_fake.backward(mone)

        g_loss = ent_fake + margin_ent_fake
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_optim.step()

        # Add FID score...
        if self.config.fid:
          with torch.no_grad():
            img_tensor = self.G(noise)
            img_list = [i for i in img_tensor]
            imgs_cur_epoch.extend(img_list)

        if (num_iter+1) % 100 == 0:
          print('Epoch: ({:2.0f}/{:2.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, len(dataloader), 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
      # end of epoch
      # Add FID score...
      if self.config.fid:
        fake_list = []
        real_list = []
        for i in range(min(len(imgs_cur_epoch), len(self.dataset))):
          fake_list.append(utils.dup2rgb(imgs_cur_epoch[i]))
          real_list.append(utils.dup2rgb(self.dataset[i][0]))
        fid_value = fid_score.calculate_fid_given_img_tensor(fake_list, real_list, 100, True, 2048)
        self.log['fid'].append(fid_value)
        print("-- FID score %.4f" % fid_value)

      # Report epoch training time.
      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))

      if (epoch+1) % 1 == 0:
        img = self.saveimg(noise_fixed, 'G-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    self.save_model(self.save_dir, self.config.num_epoch, self.models)
    print('-'*50)

    utils.generate_animation(self.save_dir, generated_images)
    utils.plot_loss(self.log, self.save_dir)    

    # Add FID score...
    if self.config.fid:
      self.plot_fid()
    
  def build_model(self):
    channel, height, width = self.dataset[0][0].size()
    assert height == width, "Image must be square."
    import models.mnist as nets
    self.G = nets.Generator(self.z_dim, channel)
    self.D = nets.CatD(channel, self.cat_dim)
    networks = [self.G, self.D]
    for i in networks:
      i.apply(utils.weights_init)
      i.to(self.device)
    return networks

  def plot_fid(self):
    plt.title('FID score')
    plt.plot(self.log['fid'], linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('FID')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(self.save_dir + '/fid.png')
    plt.close('all')

  def saveimg(self, noise, fname):
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