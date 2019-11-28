import os
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import *

class Trainer:
  def __init__(self, config, dataset, FG, G, GT, FD, D, Q):
    self.config = config
    self.dataset = dataset

    self.FG = FG
    self.G = G 
    self.GT = GT
    self.FD = FD
    self.D = D
    self.Q = Q

  def _sample(self, dis_c, con_c, noise, prob):
    onehot, idx = Noiser.Category(prob.squeeze(), self.config.batch_size)
    dis_c.data.copy_(torch.Tensor(onehot))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
    return z, idx

  def _fix_noise(self):
    'Generate fix noise for image generating during the whole training.'
    c = np.linspace(-1, 1, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])

    idx = np.arange(10).repeat(10)
    one_hot = np.zeros((100, 10))
    one_hot[range(100), idx] = 1
    fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)
    
    return fix_noise.numpy(), one_hot, c1, c2

  def _save_results(self, noise, experiment_tag: str, filename: str):
    basepath = os.getcwd()
    if self.config.dataset == 'mnist':
      savepath = os.path.join(basepath, 'results/MNIST/')
    elif self.config.dataset == 'stl10':
      savepath = os.path.join(basepath, 'results/STL10/')
    else:
      raise NotImplementedError
    
    # Add a tag.
    savepath = os.path.join(savepath, experiment_tag)
    if not os.path.exists(savepath):
      os.makedirs(savepath)
    
    img = self.G(self.FG(noise))
    save_image(img, os.path.join(savepath, filename), nrow=10)

  def train(self, cat_prob: np.ndarray, true_dist: np.ndarray):
    """
    @param cat_prob: initial categorical prob.
    @param true_dist: the true distribution of data.
    """
    np.set_printoptions(precision=4)
    # Priori configurations.
    dis_c = torch.FloatTensor(self.config.batch_size, self.config.num_class).to(self.config.device)
    con_c = torch.FloatTensor(self.config.batch_size, 2).to(self.config.device)
    noise = torch.FloatTensor(self.config.batch_size, self.config.num_noise_dim).to(self.config.device)

    for i in [dis_c, con_c, noise]:
      i.requires_grad_(True)

    criterionD = nn.BCELoss().to(self.config.device)
    criterionQ_dis = nn.CrossEntropyLoss().to(self.config.device)
    criterionQ_con = LogGaussian()

    optimD = optim.Adam([{'params': self.FD.parameters()}, {'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params': self.FG.parameters()}, {'params': self.G.parameters()}, {'params': self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

    dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=1)

    # Fixed random variables.
    fix_noise, cdis, c1, c2 = self._fix_noise()

    # cat_prob = np.random.rand(self.config.num_class)
    # cat_prob = np.array([9,7,4,1,1,1,1,4,7,9.0])
    print('Original cat dist is {}'.format(cat_prob))

    # Training...
    for epoch in range(self.config.num_epoch):
      real_labels = torch.ones(self.config.batch_size, 1).to(self.config.device)
      fake_labels = torch.zeros(self.config.batch_size, 1).to(self.config.device)
      for num_iter, (images, _) in enumerate(dataloader):
        # Train discriminator.
        ## real part
        optimD.zero_grad()

        images = images.to(self.config.device)
        images.requires_grad = True

        feout1 = self.FD(images)
        probs_real = self.D(feout1)

        # For last batch is not complete.
        if probs_real.size(0) < real_labels.size(0):
          real_labels = torch.ones(probs_real.size()).to(self.config.device)

        loss_real = criterionD(probs_real, real_labels)
        ## fake part
        z, idx = self._sample(dis_c, con_c, noise, cat_prob)

        fgout = self.FG(z)
        fake_image = self.G(fgout)

        feout2 = self.FD(fake_image.detach())
        probs_fake = self.D(feout2)

        if probs_fake.size(0) != fake_labels.size(0):
          fake_labels = torch.zeros(probs_fake.size()).to(self.config.device)

        loss_fake = criterionD(probs_fake, fake_labels)
        ## update
        D_loss = loss_real + loss_fake
        D_loss.backward()
        optimD.step()

        if num_iter % 2 == 0:
          # Train generator.
          optimG.zero_grad()

          feout = self.FD(fake_image)
          probs_fake = self.D(feout)

          if real_labels.size(0) != probs_fake.size(0):
            real_labels = torch.ones(probs_fake.size()).to(self.config.device)

          reconstruct_loss = criterionD(probs_fake, real_labels)

          q_logits, q_mu, q_var = self.Q(feout)
          classes = torch.LongTensor(idx).to(self.config.device)
          dis_loss = criterionQ_dis(q_logits, classes)
          con_loss = criterionQ_con(con_c, q_mu, q_var) * 0.1 # weight

          ## update parameters
          G_loss = reconstruct_loss + dis_loss + con_loss
          G_loss.backward()
          optimG.step()

          if self.config.use_ba:
            ## update r(c)
            trans_prob = self.GT(fgout)
            trans_prob = trans_prob.cpu().detach().numpy()
            ba = BlahutArimoto(cat_prob, trans_prob)
            cat_prob, _ = ba.Update(max(np.exp(-epoch/4), 1e-8))
        
        if num_iter % 100 == 0:
          print('Epoch/Iter: {}/{}, Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch, num_iter, D_loss.cpu().detach().numpy(), G_loss.cpu().detach().numpy())
          )

      if True:
        noise.data.copy_(torch.Tensor(fix_noise))
        dis_c.data.copy_(torch.Tensor(cdis))

        con_c.data.copy_(torch.from_numpy(c1))
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
        # x_save = self.G(self.FG(z))
        # save_image(x_save.data, os.path.join(basepath, 'c1-epoch-{}.png'.format(epoch+1)), nrow=10)
        self._save_results(z, self.config.experiment_tag, 'c1-epoch-{}.png'.format(epoch+1))

        con_c.data.copy_(torch.from_numpy(c2))
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
        # x_save = self.G(self.FG(z))
        # save_image(x_save.data, os.path.join(basepath, 'c2-epoch-{}.png'.format(epoch+1)), nrow=10)
        self._save_results(z, self.config.experiment_tag, 'c2-epoch-{}.png'.format(epoch+1))

        # TODO: This is meaningless since category code must map to the correct ground truth label first.
        rmse = np.linalg.norm(cat_prob - true_dist)
        print('Current cat dist: {}\nRMSE: {:.4f}'.format(cat_prob, rmse))
          
