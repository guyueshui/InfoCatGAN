import os
import torch
import torchvision
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms

import models.celeba as nets
from torch.utils.data import DataLoader

from utils import get_data, weights_init, ETimer, generate_animation
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
    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
      f.write(str(config.__dict__))
      # json.dump(config.__dict__, f, indent=4, sort_keys=True)
    
    self.build_model()

  def train(self):
    self.log = {}
    self.log['d_loss'] = []
    self.log['g_loss'] = []
    self.measure = {} # convergence measure
    self.measure['pre'] = []
    self.measure['pre'].append(1)
    self.measure['cur'] = []
    generated_images = []
    
    bs = self.batch_size
    dv = self.config.device
    AeLoss = nn.L1Loss().to(dv)

    z = torch.FloatTensor(bs, self.num_noise_dim).to(dv)
    z_fixed = torch.FloatTensor(bs, self.num_noise_dim).to(dv)
    z_fixed.normal_(0, 1)

    def _get_optimizer(lr):
      return optim.Adam(self.G.parameters(), lr=lr, betas=(self.config.beta1, self.config.beta2)), \
             optim.Adam(self.D.parameters(), lr=lr, betas=(self.config.beta1, self.config.beta2)),
    
    g_optim, d_optim = _get_optimizer(self.lr)
    dataloader = DataLoader(self.dataset, batch_size=bs, shuffle=True, num_workers=12)

    # Training...
    print('-'*25)
    print('Starting Training Loop...\n')
    print('Epochs: {}\nDataset: {}\nBatch size: {}\nLength of Dataloder: {}'
          .format(self.config.num_epoch, self.config.dataset, 
                  self.config.batch_size, len(dataloader)))
    print('-'*25)

    t0 = ETimer() # train timer
    t1 = ETimer() # epoch timer
    k_t = 0

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
        d_real = self.D(image)
        d_loss_real = AeLoss(d_real, image)

        fake_image = self.G(z)
        d_fake = self.D(fake_image.detach())
        d_loss_fake = AeLoss(d_fake, fake_image.detach())

        d_loss = d_loss_real - k_t * d_loss_fake
        self.log['d_loss'].append(d_loss.cpu().detach().item())
        d_loss.backward()
        d_optim.step()

        # Update generator.
        g_optim.zero_grad()
        d_fake = self.D(fake_image)
        g_loss = AeLoss(d_fake, fake_image)
        self.log['g_loss'].append(g_loss.cpu().detach().item())
        g_loss.backward()
        g_optim.step()

        # Convergence metric.
        balance = (self.gamma * d_loss_real - g_loss).item()
        temp_measure = d_loss_real + abs(balance)
        self.measure['cur'] = temp_measure.item()
        # update k_t
        k_t += self.lambda_k * balance
        k_t = max(0, min(1, k_t))

        # Print progress...
        if (num_iter+1) % 500 == 0:
          print('Epoch: ({:3.0f}/{:3.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, len(dataloader), 
          d_loss.cpu().detach().numpy(), g_loss.cpu().detach().numpy())
          )
          self.autoencode(image, self.save_dir, epoch+1, fake_image)
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
      
      if (epoch+1) % 2 == 0:
        img = self.generate(z_fixed, self.save_dir, epoch+1)
        generated_images.append(img)
        print("eposch image.device ", image.device)
        print("eposch fake_image.device ", fake_image.device)
        self.autoencode(image.to(dv), self.save_dir, epoch+1, fake_image.to(dv))

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    print('-'*50)
    generate_animation(self.save_dir, generated_images)

  
  def build_model(self):
    channel, height, width = self.dataset[0][0].size()
    assert height == width, "Height and width must equal."
    repeat_num = int(np.log2(height)) - 2
    self.G = nets.GeneratorCNN([self.batch_size, self.num_noise_dim], 
                               [3, 32, 32],
                               self.config.hidden_dim,
                               )
    self.D = nets.DiscriminatorCNN([3, 32, 32],
                                   [3, 32, 32],
                                   self.config.hidden_dim,
                                   )
    for i in [self.G, self.D]:
      i.apply(weights_init)
      i.to(self.config.device)
  
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
    img = self.D(inputs)
    vutils.save_image(img, img_path)
    if fake_inputs is not None:
      fake_img_path = os.path.join(path, 'D_fake-epoch-{}.png'.format(idx))
      fake_img = self.D(fake_inputs)
      vutils.save_image(fake_img, fake_img_path)


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
