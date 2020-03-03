import os
import json
import torch
import torchvision
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms

import models.celeba as nets
from torch.utils.data import DataLoader

from utils import get_data, weights_init
from config import get_config

class Trainer(object):
  def __init__(self, config, dataset):
    self.config = config
    self.dataset = dataset

    self.batch_size = self.config.batch_size
    self.num_noise_dim = self.config.num_noise_dim
    self.lr = self.config.lr
    self.lr_update_step = self.config.lr_update_step
    self.max_step = self.config.max_step
    self.gamma = self.config.gamma
    self.lambda_k = self.config.lambda_k

    save_dir = os.path.join(os.getcwd(), 'results', 
                            config.dataset, config.experiment_tag)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    self.save_dir = save_dir
    # Write experiment settings to file.
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
      json.dump(config.__dict__, f, indent=4, sort_keys=True)
    
    self.build_model()

  def train(self):
    bs = self.batch_size
    dv = self.config.device
    AeLoss = nn.L1Loss().to(dv)

    z = torch.FloatTensor(bs, self.num_noise_dim).to(dv)
    z_fixed = torch.FloatTensor(bs, self.num_noise_dim).to(dv)

    def _get_optimizer(lr):
      return optim.Adam(self.G.parameters(), lr=lr, betas=(self.config.beta1, self.config.beta2)), \
             optim.Adam(self.D.parameters(), lr=lr, betas=(self.config.beta1, self.config.beta2)),
    
    g_optim, d_optim = _get_optimizer(self.lr)
    dataloader = DataLoader(self.dataset, batch_size=bs, shuffle=True, num_workers=1)
    data_loader = iter(dataloader)
    x_fixed = next(data_loader)
    vutils.save_image(x_fixed, '{}/x_fixed.png'.format(self.save_dir))

    k_t = 0
    prev_measure = 1
    from collections import deque
    measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

    for step in range(self.max_step):
      try:
        image = next(data_loader)
      except StopIteration:
        data_loader = iter(dataloader)
        image = next(data_loader)

      self.D.zero_grad()
      self.G.zero_grad()

      z.data.normal_(0, 1)
      fake_image = self.G(z)
      
      ae_d_real = self.D(image)
      ae_d_fake = self.D(fake_image.detach())
      ae_g = self.D(fake_image)

      d_loss_real = AeLoss(ae_d_real, image)
      d_loss_fake = AeLoss(ae_d_fake, fake_image.detach())
      d_loss = d_loss_real - k_t * d_loss_fake

      g_loss = AeLoss(ae_g, fake_image)
      
      loss = d_loss + g_loss
      loss.backward()

      g_optim.step()
      d_optim.step()

      g_d_balance = (self.gamma * d_loss_real - d_loss_fake)
      k_t += self.lambda_k * g_d_balance
      k_t = max(min(1, k_t), 0)

      measure = d_loss_real + abs(g_d_balance)
      measure_history.append(measure)

      if step % 50 == 0:
        print("[{}/{}] Loss_D {:.4f} L_x: {:4f} Loss_G: {:.4f} "
              "measure: {:.4f}, k_t: {:4f}, lr: {:.7f}"
              .format(step, self.max_step, d_loss.detach().cpu(), d_loss_real.detach().cpu(),
                      g_loss.detach().cpu(), measure, k_t, self.lr))
      x_fake =  self.generate(z_fixed, self.save_dir, idx=step)
      self.autoencode(x_fixed, self.save_dir, idx=step, x_fake=x_fake)

      if (step+1) % self.lr_update_step == 0:
        cur_measure = np.mean(measure_history)
        if cur_measure > prev_measure * 0.9999:
          self.lr *= 0.5
          g_optim, d_optim = _get_optimizer(self.lr)
        prev_measure = cur_measure
  
  def build_model(self):
    self.G = nets.GeneratorCNN([self.batch_size, self.num_noise_dim], 
                               [3, 32, 32],
                               self.config.hidden_dim,
                               self.config.repeat_num)
    self.D = nets.DiscriminatorCNN([3, 32, 32],
                                   [3, 32, 32],
                                   self.config.hidden_dim,
                                   self.config.repeat_num)
    for i in [self.G, self.D]:
      i.apply(weights_init)
      i.to(self.config.device)

  def generate(self, inputs, path, idx=None):
    path = '{}/{}_G.png'.format(path, idx)
    x = self.G(inputs)
    vutils.save_image(x.data, path)
    print("[*] Samples saved: {}".format(path))
    return x

  def autoencode(self, inputs, path, idx=None, x_fake=None):
    x_path = '{}/{}_D.png'.format(path, idx)
    x = self.D(inputs)
    vutils.save_image(x.data, x_path)
    print("[*] Samples saved: {}".format(x_path))

    if x_fake is not None:
      x_fake_path = '{}/{}_D_fake.png'.format(path, idx)
      x = self.D(x_fake)
      vutils.save_image(x.data, x_fake_path)
      print("[*] Samples saved: {}".format(x_fake_path))
  

def main(config):
  assert config.dataset == 'CelebA', "CelebA support only."
  dataset = get_data('CelebA', config.data_root)
  print("this is celeba dataset, len ", len(dataset))

  if config.gpu == 0:  # GPU selection.
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  elif config.gpu == 1:
    config.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
  elif config.gpu == -1:
    config.device = torch.device('cpu')
  else:
    raise IndexError('Invalid GPU index')

  t = Trainer(config, dataset)
  t.train()


if __name__ == '__main__':
  args = get_config()
  print(args)
  main(args)
