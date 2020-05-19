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
    self.log['fid'] = []
    self.log['ent'] = []
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
    #QcontLoss = nn.MSELoss().to(dv)
    QcontLoss = utils.LogGaussian()
    mseloss = nn.MSELoss().to(dv)

    g_step_params = [{'params': self.G.parameters()}, {'params': self.Q.parameters()}]
    d_step_params = [{'params': self.FD.parameters()}, {'params': self.D.parameters()}]
    g_optim = optim.Adam(g_step_params, lr=1e-3, betas=(0.5, 0.99))
    d_optim = optim.Adam(d_step_params, lr=2e-4, betas=(0.5, 0.99))
      
    dataloader = DataLoader(self.dataset, batch_size=bs, shuffle=True, num_workers=1)
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

      # Add FID score...
      if self.config.fid:
        imgs_cur_epoch = []

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
        fmap_real = self.FD(image)
        d_real = self.D(fmap_real)
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
        q_loss_disc = QdiscLoss(q_logits, targets)
        q_loss_conc = QcontLoss(cont_c, mu, var) * 0.1
        self.log['ent'].append( (q_loss_disc + q_loss_conc).cpu().detach().item() )

        # feature matching loss
        fmatch_loss = mseloss(d_body_out, fmap_real.detach())

        g_loss = reconstruct_loss + q_loss_disc + q_loss_conc + fmatch_loss*0.9
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
        fid_value = utils.ComputeFID(imgs_cur_epoch, self.dataset)
        self.log['fid'].append(fid_value)
        print("-- FID score %.4f" % fid_value)

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

    np.savez(self.save_dir + '/numbers.npz',
             ent=self.log['ent'],
             g_loss=self.log['g_loss'],
             d_loss=self.log['d_loss'])
    plt.title('MI loss')
    plt.plot(self.log['ent'])
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.savefig(self.save_dir + '/ent_loss.pdf')

    # Manipulating continuous latent codes.
    self.generate(self.G, fixed_noise_1, 'c1-final.png')
    self.generate(self.G, fixed_noise_2, 'c2-final.png')

    utils.generate_animation(self.save_dir, generated_images)
    utils.plot_loss(self.log, self.save_dir)    
  
  def build_model(self):
    channel, height, width = self.dataset[0][0].size()
    assert height == width, "Height and width must equal."
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

  def semi_train(self, num_labels=100):
    self.log = {}
    self.log['d_loss'] = []
    self.log['g_loss'] = []
    self.log['fid'] = []
    self.log['ent'] = []
    generated_images = []

    bs = self.config.batch_size
    dv = self.device
    supervised_ratio = num_labels / len(self.dataset)

    z = torch.FloatTensor(bs, self.z_dim).to(dv)
    disc_c = torch.FloatTensor(bs, self.cat_dim*self.num_disc_code).to(dv)
    cont_c = torch.FloatTensor(bs, self.num_cont_code).to(dv)
    real_label = 1
    fake_label = 0

    bceloss = nn.BCELoss().to(dv)
    celoss = nn.CrossEntropyLoss().to(dv)
    gaussian = utils.LogGaussian()
    mseloss = nn.MSELoss().to(dv)
    
    d_param_group = [
      {'params': self.FD.parameters()},
      {'params': self.D.parameters()},
    ]
    g_param_group = [
      {'params': self.G.parameters()},
      {'params': self.Q.parameters()},
    ]
    d_optim = optim.Adam(d_param_group, lr=2e-4, betas=(0.5, 0.99))
    g_optim = optim.Adam(g_param_group, lr=1e-3, betas=(0.5, 0.99))
    q_optim = optim.Adam(self.Q.parameters(), lr=1e-3, betas=(0.5, 0.99))

    dset = utils.CustomDataset(self.dataset, supervised_ratio)
    dset.report()
    labeled_loader = DataLoader(dset.labeled, batch_size=bs, shuffle=True, num_workers=1)
    labeled_iter = iter(labeled_loader)
    unlabeled_loader = DataLoader(dset.unlabeled, batch_size=bs, shuffle=True, num_workers=1)
    unlabeled_iter = iter(unlabeled_loader)

    fixz, cdis, c1, c2, c0 = self.generate_fix_noise()
    fixed_noise = np.hstack([fixz, cdis, c0])
    fixed_noise_1 = np.hstack([fixz, cdis, c1])
    fixed_noise_2 = np.hstack([fixz, cdis, c2])
    # NOTE: dtype should exactly match the network weight's type!
    fixed_noise = torch.as_tensor(fixed_noise, dtype=torch.float32).to(dv)
    fixed_noise_1 = torch.as_tensor(fixed_noise_1, dtype=torch.float32).to(dv)
    fixed_noise_2 = torch.as_tensor(fixed_noise_2, dtype=torch.float32).to(dv)

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

    for epoch in range(self.config.num_epoch):
      t1.reset()
      num_labeled_batch = 0
      if supervised_prob >= 0.1:
        tot_iters = 100
      else:
        tot_iters = len(unlabeled_loader) + len(labeled_loader)
      
      # Add FID score...
      if self.config.fid:
        imgs_cur_epoch = []

      self.FD.train()
      self.D.train()
      for num_iter in range(tot_iters):

        self.G.train()
        self.Q.train()
        # Train discriminator.
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
          print("Data consumed, restart iter...")
          print(not cur_batch)

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
        
        dbody_out_real = self.FD(image)
        prob_real = self.D(dbody_out_real)
        label = torch.full_like(prob_real, real_label, device=dv)
        loss_real = bceloss(prob_real, label)
        loss_real.backward()

        # update Q to bind real label if cur_batch is_labeled_batch
        if is_labeled_batch:
          q_optim.zero_grad()
          disc_logits_real, _, _ = self.Q(dbody_out_real.detach())
          qsuper_loss_real = celoss(disc_logits_real, y)
          qsuper_loss_real.backward()
          q_optim.step()

        if is_labeled_batch:
          idx = y.cpu().numpy()
          z.normal_(0, 1)
          onehot = torch.zeros_like(disc_c)
          onehot[range(bs), idx] = 1
          disc_c.copy_(onehot)
          cont_c.uniform_(-1, 1)
          noise = torch.cat([z, disc_c, cont_c], dim=1).view(bs, -1)
        else:
          noise, idx = self.generate_noise(z, disc_c, cont_c)

        fake_image = self.G(noise)
        ## Add instance noise if specified.
        if self.config.instance_noise:
          # instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise
        dbody_out_fake = self.FD(fake_image.detach())
        prob_fake = self.D(dbody_out_fake)
        label.fill_(fake_label)
        loss_fake = bceloss(prob_fake, label)
        loss_fake.backward()

        d_loss = loss_real + loss_fake
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        d_optim.step()

        # Train generator.
        g_optim.zero_grad()
        # NOTE: why bother to get a new output of FD? cause FD is updated by optimizing Discriminator.
        dbody_out = self.FD(fake_image)
        prob_fake = self.D(dbody_out)
        label.fill_(real_label)
        reconstruct_loss = bceloss(prob_fake, label)

        q_logits, q_mu, q_var = self.Q(dbody_out)
        targets = torch.LongTensor(idx).to(dv)
        if is_labeled_batch:
          dis_loss = celoss(q_logits, targets) * 1.2
        else:
          dis_loss = celoss(q_logits, targets) * 1.0
        con_loss = gaussian(cont_c, q_mu, q_var) * 0.2
        self.log['ent'].append( (dis_loss + con_loss).cpu().detach().item() )

        # feature matching loss
        fmatch_loss = mseloss(dbody_out, dbody_out_real.detach())

        g_loss = reconstruct_loss + dis_loss + con_loss + fmatch_loss*1.1
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
        g_optim.step()

        # Add FID score...
        if self.config.fid and tot_iters > 200:
          with torch.no_grad():
            img_tensor = self.G(noise)
            img_list = [i for i in img_tensor]
            imgs_cur_epoch.extend(img_list)
            # print("imgs_cur_epoch.len ", len(imgs_cur_epoch))
            # print(imgs_cur_epoch[0].size())

        # Print progress...
        if (num_iter+1) % 50 == 0:
          print('Epoch: ({:2.0f}/{:2.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, tot_iters, 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
      
      # Add FID score...
      if self.config.fid and (epoch+1) == self.config.num_epoch:
        fid_value = utils.ComputeFID(imgs_cur_epoch, self.dataset)
        self.log['fid'].append(fid_value)
        print("-- FID score %.4f" % fid_value)

      if (epoch+1) % 25 == 0:
        self.save_model(self.save_dir, epoch+1, *self.models)

      # Report epoch training time.
      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))
      print('labeled batch for Epoch %d: %d/%d' % (epoch+1, num_labeled_batch, tot_iters))
      if supervised_prob > 0.01:
        supervised_prob -= 0.1
      if supervised_prob < 0.01:
        supervised_prob = 0.01

      if (epoch+1) % 1 == 0:
        img = self.generate(self.G, fixed_noise_1, 'c0-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)

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
    np.savez(self.save_dir + '/numbers.npz',
             ent=self.log['ent'],
             g_loss=self.log['g_loss'],
             d_loss=self.log['d_loss'])