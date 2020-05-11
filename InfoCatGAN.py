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

class InfoCatGAN(utils.BaseModel):
  def __init__(self, config, dataset):
    super(InfoCatGAN, self).__init__(config, dataset)
    self.z_dim = config.num_noise_dim
    self.cat_dim = 10

    self.models = self.build_model()

  def train(self):
    self.log = {}
    self.log['d_loss'] = []
    self.log['g_loss'] = []
    self.log['fid'] = []
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
      g_step_params = [{'params': self.G.parameters()}]#, {'params': self.D.parameters()}]
      d_step_params = [{'params': self.D.parameters()}]
      return optim.Adam(g_step_params, lr=0.001, betas=(self.config.beta1, self.config.beta2)), \
             optim.Adam(d_step_params, lr=0.0002, betas=(self.config.beta1, self.config.beta2)),
      
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

    self.D.train()
    for epoch in range(self.config.num_epoch):
      # Add FID score...
      if self.config.fid:
        imgs_cur_epoch = []

      self.G.train()
      t1.reset()
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
          std = -0.1 / tot_iters * num_iter + 0.03
          instance_noise.normal_(0, std)
          image = image + instance_noise

        d_real_simplex, _ = self.D(image)
        # Minimize entropy to make certain prediction of real sample.
        ent_real = utils.Entropy(d_real_simplex)
        # Maximize marginal entropy over real samples to ensure equal usage.
        margin_ent_real = utils.MarginalEntropy(d_real_simplex)

        noise, idx = self.generate_noise(z, disc_c)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

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
        binding_loss = celoss(d_fake_logits, targets) * 0.03

        g_loss = ent_fake - margin_ent_fake + binding_loss
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
        g_optim.step()

        # Add FID score...
        if self.config.fid:
          with torch.no_grad():
            img_tensor = self.G(noise)
            img_list = [i for i in img_tensor]
            imgs_cur_epoch.extend(img_list)

        # Print progress...
        if (num_iter+1) % 100 == 0:
          print('Epoch: ({:2.0f}/{:2.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, len(dataloader), 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
      # end of epoch

      # Add FID score...
      if self.config.fid and (epoch+1) == self.config.num_epoch:
        fake_list = []
        real_list = []
        for i in range(min(len(imgs_cur_epoch), len(self.dataset))):
          fake_list.append(utils.dup2rgb(imgs_cur_epoch[i]))
          real_list.append(utils.dup2rgb(self.dataset[i][0]))
        fid_value = fid_score.calculate_fid_given_img_tensor(fake_list, real_list, 100, True, 2048)
        self.log['fid'].append(fid_value)
        print("-- FID score %.4f" % fid_value)

      if (epoch+1) % 25 == 0:
        self.save_model(self.save_dir, epoch+1, *self.models)

      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))

      if (epoch+1) % 1 == 0:
        img = self.generate(self.G, noise_fixed, 'G-epoch-{}.png'.format(epoch+1))
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
    self.G = nets.OfficialGenerator(noise_dim, channel)
    self.D = nets.OfficialCatD(channel, self.cat_dim)
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

  def raw_classify(self, imgs):
    imgs = imgs.to(self.device)
    self.D.eval()
    with torch.no_grad():
      _, logits = self.D(imgs)
    return logits.cpu().numpy()

  def semi_train(self):
    self.log = {}
    self.log['d_loss'] = []
    self.log['g_loss'] = []
    self.log['fid'] = []
    generated_images = []
    
    bs = self.config.batch_size
    dv = self.device
    supervised_ratio = 132 / len(self.dataset)

    z = torch.FloatTensor(bs, self.z_dim).to(dv)
    disc_c = torch.FloatTensor(bs, self.cat_dim).to(dv)
    # Fixed random variables.
    fixz, cdis = self.generate_fix_noise()
    noise_fixed = np.hstack([fixz, cdis])
    noise_fixed = torch.as_tensor(noise_fixed, dtype=torch.float32).to(dv)

    celoss = nn.CrossEntropyLoss().to(dv)

    def _get_optimizer(lr):
      g_step_params = [{'params': self.G.parameters()}]#, {'params': self.D.parameters()}]
      d_step_params = [{'params': self.D.parameters()}]
      return optim.Adam(g_step_params, lr=0.001, betas=(self.config.beta1, self.config.beta2)), \
             optim.Adam(d_step_params, lr=0.0002, betas=(self.config.beta1, self.config.beta2)),
      
    g_optim, d_optim = _get_optimizer(0)

    dset = utils.CustomDataset(self.dataset, supervised_ratio)
    dset.report()
    labeled_loader = DataLoader(dset.labeled, batch_size=bs, shuffle=True, num_workers=1)
    labeled_iter = iter(labeled_loader)
    unlabeled_loader = DataLoader(dset.unlabeled, batch_size=bs, shuffle=True, num_workers=1)
    unlabeled_iter = iter(unlabeled_loader)

    # Training...
    print('-'*25)
    print('Starting Training Loop...\n')
    print('Epochs: {}\nDataset: {}\nBatch size: {}\nLength of Dataloder: {}'
          .format(self.config.num_epoch, self.config.dataset, 
                  self.config.batch_size, None))
    print('-'*25)

    t0 = utils.ETimer() # train timer
    t1 = utils.ETimer() # epoch timer
    supervised_prob = 0.99

    self.D.train()
    for epoch in range(self.config.num_epoch):
      # Add FID score...
      if self.config.fid:
        imgs_cur_epoch = []

      t1.reset()

      num_labeled_batch = 0
      if supervised_prob >= 0.1:
        tot_iters = 100
      else:
        tot_iters = len(labeled_loader) + len(unlabeled_loader)

      self.G.train()
      for num_iter in range(tot_iters):

        # Update discriminator.
        d_optim.zero_grad()
        cur_batch = None
        # Biased coin toss to decide if we sampling from labeled or unlabeled data.
        is_labeled_batch = (torch.bernoulli(torch.tensor(supervised_prob)) == 1)
        try:
          if is_labeled_batch:
            cur_batch = next(labeled_iter)
            num_labeled_batch += 1
          else:
            cur_batch = next(unlabeled_iter)
        except StopIteration:
          print(num_iter, "restart batch")

        if not cur_batch or cur_batch[0].size(0) != bs:
          if is_labeled_batch:
            labeled_iter = iter(labeled_loader)
            cur_batch = next(labeled_iter)
          else:
            unlabeled_iter = iter(unlabeled_loader)
            cur_batch = next(unlabeled_iter)

        image, y = [e.to(dv) for e in cur_batch]

        # Guided by https://github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        # This may be helpful for generator convergence.
        if self.config.instance_noise:
          instance_noise = torch.zeros_like(image)
          std = -0.1 / tot_iters * num_iter + 0.1
          instance_noise.normal_(0, std)
          image = image + instance_noise

        d_real_simplex, real_logits = self.D(image)
        if is_labeled_batch:
          bind_loss = celoss(real_logits, y) * 1.1
        else:
          bind_loss = 0.0
        # Minimize entropy to make certain prediction of real sample.
        ent_real = utils.Entropy(d_real_simplex)
        # Maximize marginal entropy over real samples to ensure equal usage.
        margin_ent_real = utils.MarginalEntropy(d_real_simplex)

        noise, idx = self.generate_noise(z, disc_c)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

        d_fake_simplex, _ = self.D(fake_image.detach())
        # Maximize entropy to make uncertain prediction of fake sample.
        ent_fake = utils.Entropy(d_fake_simplex)

        d_loss = ent_real - margin_ent_real - ent_fake + bind_loss
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
        binding_loss = celoss(d_fake_logits, targets) * 0.03

        g_loss = ent_fake - margin_ent_fake + binding_loss
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
        g_optim.step()

        # Add FID score...
        if self.config.fid:
          with torch.no_grad():
            img_tensor = self.G(noise)
            img_list = [i for i in img_tensor]
            imgs_cur_epoch.extend(img_list)

        # Print progress...
        if (num_iter+1) % 100 == 0:
          print('Epoch: ({:2.0f}/{:2.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, tot_iters, 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
      # end of epoch
      if (epoch+1) % 25 == 0:
        self.save_model(self.save_dir, epoch+1, *self.models)
      # Add FID score...
      if self.config.fid and (epoch+1) == self.config.num_epoch:
        fake_list = []
        real_list = []
        for i in range(min(len(imgs_cur_epoch), len(self.dataset))):
          fake_list.append(utils.dup2rgb(imgs_cur_epoch[i]))
          real_list.append(utils.dup2rgb(self.dataset[i][0]))
        fid_value = fid_score.calculate_fid_given_img_tensor(fake_list, real_list, 100, True, 2048)
        self.log['fid'].append(fid_value)
        print("-- FID score %.4f" % fid_value)

      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))
      print('labeled batch for Epoch %d: %d/%d' % (epoch+1, num_labeled_batch, tot_iters))
      if supervised_prob > 0.01:
        supervised_prob -= 0.1
      if supervised_prob < 0.01:
        supervised_prob = 0.01

      if (epoch+1) % 1 == 0:
        img = self.generate(self.G, noise_fixed, 'G-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    self.save_model(self.save_dir, self.config.num_epoch, *self.models)
    print('-'*50)

    utils.generate_animation(self.save_dir, generated_images)
    utils.plot_loss(self.log, self.save_dir)    