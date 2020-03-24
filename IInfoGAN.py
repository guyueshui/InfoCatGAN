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
from torch.utils.data import DataLoader

import utils
import models.mnist as nets

class IInfoGAN(utils.BaseModel):
  def __init__(self, config, dataset):
    super(IInfoGAN, self).__init__(config, dataset)
    self.z_dim = config.num_noise_dim
    self.c_dim = config.num_con_c

    self.lr = self.config.lr
    self.gamma = self.config.gamma
    self.lambda_k = self.config.lambda_k

    self.models = self.build_model()

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
    
    bs = self.config.batch_size
    dv = self.device
    real_label = 1
    fake_label = 0
    alpha = 1.0 # prefer BEGAN

    z = torch.FloatTensor(bs, self.z_dim).to(dv)
    c = torch.FloatTensor(bs, self.c_dim).to(dv)
    labels = torch.FloatTensor(bs, 1).to(dv)
    # Fixed nosies
    fixz, c1, c2, c0 = self.generate_fix_noise()
    noise_fixed = np.hstack([fixz, c0])
    fixed_noise_1 = np.hstack([fixz, c1])
    fixed_noise_2 = np.hstack([fixz, c2])
    # NOTE: dtype should exactly match the network weight's type!
    noise_fixed = torch.as_tensor(noise_fixed, dtype=torch.float32).to(dv)
    fixed_noise_1 = torch.as_tensor(fixed_noise_1, dtype=torch.float32).to(dv)
    fixed_noise_2 = torch.as_tensor(fixed_noise_2, dtype=torch.float32).to(dv)
    
    AeLoss = nn.L1Loss().to(dv)
    MSELoss = nn.MSELoss().to(dv)
    BCELoss = nn.BCELoss().to(dv)

    def _get_optimizer(lr):
      d_lr = lr
      g_lr = d_lr * 5
      g_parameters = [{'params':self.G.parameters()}, {'params':self.Q.parameters()}]
      d_parameters = [{'params':self.D.parameters()}, {'params':self.DProb.parameters()}]
      return optim.Adam(g_parameters, lr=g_lr, betas=(self.config.beta1, self.config.beta2)), \
             optim.Adam(d_parameters, lr=d_lr, betas=(self.config.beta1, self.config.beta2)),
    
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
    self.Q.train()
    self.DProb.train()
    for epoch in range(self.config.num_epoch):
      self.G.train()
      t1.reset()
      for num_iter, (image, _) in enumerate(dataloader):
        if image.size(0) != bs:
          break
        
        image = image.to(dv)
        noise = self.generate_noise(z, c)

        # Update discriminator.
        d_optim.zero_grad()
        code, d_real = self.D(image)
        prob_real = self.DProb(code)
        labels.fill_(real_label)

        d_probloss_real = BCELoss(prob_real, labels)
        d_loss_real = AeLoss(d_real, image)
        real_part = d_loss_real * alpha + d_probloss_real * (1-alpha)
        real_part.backward()

        # fake part
        fake_image = self.G(noise)
        code, d_fake = self.D(fake_image.detach())
        prob_fake = self.DProb(code)
        labels.fill_(fake_label)

        d_probloss_fake = BCELoss(prob_fake, labels)
        d_loss_fake = AeLoss(d_fake, fake_image.detach())
        fake_part = d_loss_fake * (-k_t*alpha) + d_probloss_fake * (1-alpha)
        fake_part.backward()

        d_loss = real_part + fake_part
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        # d_loss.backward()
        d_optim.step()

        # Update generator.
        g_optim.zero_grad()
        code, d_fake = self.D(fake_image)
        prob_fake = self.DProb(code)
        labels.fill_(real_label)
        reconstruct_loss = BCELoss(prob_fake, labels)

        q_c = self.Q(code)
        q_loss = MSELoss(c, q_c)
        loss_g = AeLoss(d_fake, fake_image)
        g_loss = (reconstruct_loss + q_loss)*(1-alpha) + loss_g*alpha

        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
        g_optim.step()

        # Convergence metric.
        balance = (self.gamma * d_loss_real - loss_g).item()
        temp_measure = d_loss_real + abs(balance)
        self.measure['cur'] = temp_measure.item()
        self.log['measure'].append(self.measure['cur'])
        # update k_t
        k_t += self.lambda_k * balance
        k_t = max(0, min(1, k_t))
        self.log['k_t'].append(k_t)

        # Print progress...
        if (num_iter+1) % 100 == 0:
          print('Epoch: ({:2.0f}/{:2.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, len(dataloader), 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
          self.generate(noise_fixed, 'G-epoch-{}.png'.format(epoch+1))
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
        img = self.generate(noise_fixed, 'G-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)
        self.autoencode(real_fixed_image, self.save_dir, epoch+1, fake_image)

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    self.save_model(self.save_dir, self.config.num_epoch, *self.models)
    print('-'*50)

    # Manipulating continuous latent codes.
    self.generate(fixed_noise_1, 'c1-final.png')
    self.generate(fixed_noise_2, 'c2-final.png')

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
    noise_dim = self.z_dim + self.c_dim
    hidden_dim = self.config.hidden_dim
    self.G = nets.GeneratorCNN(noise_dim, channel, hidden_dim, repeat_num)
    self.D = nets.DiscriminatorCNN(channel, channel, hidden_dim, repeat_num)
    self.DProb = nets.DProb()
    self.Q = nets.Qhead(latent_dim, self.c_dim)
    for i in [self.G, self.D, self.DProb, self.Q]:
      i.apply(utils.weights_init)
      i.to(self.device)
      utils.print_network(i)
    return self.G, self.D, self.DProb, self.Q

  def generate_noise(self, z, c):
    'Generate samples for G\'s input.'
    batch_size = z.size(0)
    z.normal_(0, 1)
    c.uniform_(-1, 1)
    noise = torch.cat([z, c], dim=1).view(batch_size, -1)
    return noise

  def generate_fix_noise(self):
    'Generate fix noise for image generating during the whole training.'
    fixz = torch.Tensor(100, self.z_dim).normal_(0, 1.0)

    c = np.linspace(-1, 1, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])
    c3 = np.random.uniform(-1, 1, (100, self.c_dim))

    return fixz.numpy(), c1, c2, c3
  
  def generate(self, noise, fname):
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

  def autoencode(self, inputs, path, idx=None, fake_inputs=None):
    img_path = os.path.join(path, 'D-epoch-{}.png'.format(idx))
    _, img = self.D(inputs)
    vutils.save_image(img, img_path, nrow=10)
    if fake_inputs is not None:
      fake_img_path = os.path.join(path, 'D_fake-epoch-{}.png'.format(idx))
      _, fake_img = self.D(fake_inputs)
      vutils.save_image(fake_img, fake_img_path, nrow=10)
  
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