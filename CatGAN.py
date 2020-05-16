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
    # one = torch.FloatTensor([1]).to(dv)
    # mone = one * -1
    # balance_set = utils.CustomDataset(self.dataset, 200/len(self.dataset))
    # balance_set.report()
    # balance_set = balance_set.labeled
    # loader = DataLoader(balance_set, batch_size=len(balance_set))
    # train_base, _ = next(iter(loader))
    # train_base = train_base.to(dv)
    
    g_optim = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    d_optim = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.9))
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
      self.G.train()
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

        d_real_simplex, _ = self.D(image)
        # Minimize entropy to make certain prediction of real sample.
        ent_real = utils.Entropy(d_real_simplex)
        # Maximize marginal entropy over real samples to ensure equal usage.
        margin_ent_real = utils.MarginalEntropy(d_real_simplex)

        noise = torch.randn(bs, self.z_dim).to(dv)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

        d_fake_simplex, _ = self.D(fake_image.detach())
        # Maximize entropy to make uncertain prediction of fake sample.
        ent_fake = utils.Entropy(d_fake_simplex)

        d_loss = ent_real - ent_fake - margin_ent_real
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        d_loss.backward()
        d_optim.step()

        # Train generator.
        g_optim.zero_grad()

        # Does noise need to be regenerated? No!
        # noise = torch.randn(bs, self.z_dim).to(dv)
        # fake_image = self.G(noise)

        # ## Add instance noise if specified.
        # if self.config.instance_noise:
        #   # instance_noise.normal_(0, std)  # Regenerate instance noise!
        #   fake_image = fake_image + instance_noise

        d_fake_simplex, _ = self.D(fake_image)
        # Fool D to make it believe the fake is real.
        ent_fake = utils.Entropy(d_fake_simplex)
        # Ensure equal usage of fake samples.
        margin_ent_fake = utils.MarginalEntropy(d_fake_simplex)

        g_loss = ent_fake - margin_ent_fake
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
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
      if (epoch+1) % self.config.save_epoch == 0:
        self.save_model(self.save_dir, epoch+1, *self.models)

      # Compute FID score...
      if self.config.fid and (epoch+1) == self.config.num_epoch:
        fid_value = utils.ComputeFID(imgs_cur_epoch, self.dataset)
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
    self.save_model(self.save_dir, self.config.num_epoch, *self.models)
    print('-'*50)

    utils.generate_animation(self.save_dir, generated_images)
    utils.plot_loss(self.log, self.save_dir)    

    # Add FID score...
    if self.config.fid:
      self.plot_fid()
    
  def build_model(self):
    channel, height, width = self.dataset[0][0].size()
    assert height == width, "Image must be square."
    import models.cifar10 as nets
    self.G = nets.CatG(self.z_dim, channel)
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

  def raw_classify(self, imgs):
    imgs = imgs.to(self.device)
    self.D.eval()
    with torch.no_grad():
      _, logits = self.D(imgs)
    return logits.cpu().numpy()

  def semi_train(self, num_labels=100):
    self.log = {}
    self.log['d_loss'] = []
    self.log['g_loss'] = []
    self.log['fid'] = []
    generated_images = []

    bs = self.config.batch_size
    dv = self.device
    supervised_ratio = num_labels / len(self.dataset)
    celoss = nn.CrossEntropyLoss().to(dv)

    dset = utils.CustomDataset(self.dataset, supervised_ratio)
    dset.report()
    labeled_loader = DataLoader(dset.labeled, batch_size=bs, shuffle=True, num_workers=1)
    labeled_iter = iter(labeled_loader)
    unlabeled_loader = DataLoader(dset.unlabeled, batch_size=bs, shuffle=True, num_workers=1)
    unlabeled_iter = iter(unlabeled_loader)
    
    g_optim = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    d_optim = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.9))

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
    noise_fixed = torch.randn(100, self.z_dim).to(dv)

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

        noise = torch.randn(bs, self.z_dim).to(dv)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

        d_fake_simplex, _ = self.D(fake_image.detach())
        # Maximize entropy to make uncertain prediction of fake sample.
        ent_fake = utils.Entropy(d_fake_simplex)

        d_loss = ent_real - ent_fake + bind_loss - margin_ent_real
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        d_loss.backward()
        d_optim.step()

        # Train generator.
        g_optim.zero_grad()
        d_fake_simplex, _ = self.D(fake_image)
        # Fool D to make it believe the fake is real.
        ent_fake = utils.Entropy(d_fake_simplex)
        # Ensure equal usage of fake samples.
        margin_ent_fake = utils.MarginalEntropy(d_fake_simplex)

        g_loss = ent_fake - margin_ent_fake
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
        g_optim.step()

        # Add FID score...
        if self.config.fid:
          with torch.no_grad():
            img_tensor = self.G(noise)
            img_list = [i for i in img_tensor]
            imgs_cur_epoch.extend(img_list)

        if (num_iter+1) % 100 == 0:
          print('Epoch: ({:2.0f}/{:2.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, tot_iters, 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
      # end of epoch
      if (epoch+1) % 25 == 0:
        self.save_model(self.save_dir, epoch+1, *self.models)

      # Compute FID score...
      if self.config.fid and (epoch+1) == self.config.num_epoch:
        fid_value = utils.ComputeFID(imgs_cur_epoch, self.dataset)
        self.log['fid'].append(fid_value)
        print("-- FID score %.4f" % fid_value)

      # Report epoch training time.
      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))
      print('labeled batch for Epoch %d: %d/%d' % (epoch+1, num_labeled_batch, tot_iters))
      if supervised_prob > 0.01:
        supervised_prob -= 0.1
      if supervised_prob < 0.01:
        supervised_prob = 0.01

      if (epoch+1) % 1 == 0:
        img = self.saveimg(noise_fixed, 'G-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    self.save_model(self.save_dir, self.config.num_epoch, *self.models)
    print('-'*50)

    utils.generate_animation(self.save_dir, generated_images)
    utils.plot_loss(self.log, self.save_dir)    