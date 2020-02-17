import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from utils import BlahutArimoto, Noiser, LogGaussian, ETimer, generate_animation

torch.set_default_tensor_type(torch.FloatTensor)

class Trainer:
  def __init__(self, config, dataset, G, FD, D, Q):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    self.config = config
    self.dataset = dataset

    self.G = G 
    self.FD = FD
    self.D = D
    self.Q = Q

    # from models.official_mnist import Qsemi
    # self.Qsemi = Qsemi().to(self.config.device)

    savepath = os.path.join(os.getcwd(), 'results', 
                            config.dataset, config.experiment_tag)
    if not os.path.exists(savepath):
      os.makedirs(savepath)
    
    self._savepath = savepath
    # Write experiment settings to file.
    with open(os.path.join(savepath, 'config.txt'), 'w') as f:
      f.write(str(config.__dict__))


  def _sample(self, z, dis_c, con_c, prob, batch_size):
    'Generate samples for G\'s input.'
    onehot, idx = Noiser.Category(prob.squeeze(), batch_size)
    z.normal_(0, 1)
    dis_c.copy_(torch.tensor(onehot))
    con_c.uniform_(-1, 1)
    noise = torch.cat([z, dis_c, con_c], dim=1).view(batch_size, -1, 1, 1)
    return noise, idx

  def _fix_noise(self):
    'Generate fix noise for image generating during the whole training.'
    fixz = torch.Tensor(100, self.config.num_noise_dim).normal_(0, 1.0)

    c = np.linspace(-1, 1, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])
    c3 = np.random.uniform(-1, 1, (100, 2))

    idx = np.arange(self.config.num_class).repeat(10)
    one_hot = np.zeros((100, self.config.num_class))
    one_hot[range(100), idx] = 1
    
    return fixz.numpy(), one_hot, c1, c2, c3

  def _save_image(self, noise, fname: str):
    from PIL import Image
    from torchvision.utils import make_grid
    tensor = self.G(noise)
    grid = make_grid(tensor, nrow=10, padding=2)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(os.path.join(self._savepath, fname))
    return ndarr

  def _save_checkpoint(self, fname: str):
    savepath = os.path.join(self._savepath, 'checkpoint')
    if not os.path.exists(savepath):
      os.makedirs(savepath)
    torch.save(
      {
        'Generator': self.G.state_dict(),
        'FrontD': self.FD.state_dict(),
        'Discriminator': self.D.state_dict(),
        'Q': self.Q.state_dict(),
        'params': self.config.__dict__
      },
      os.path.join(savepath, fname)
    )


  def train(self, cat_prob: np.ndarray, true_dist: np.ndarray):
    """
    @param cat_prob: initial categorical prob.
    @param true_dist: the true distribution of data.
    """
    np.set_printoptions(precision=4)
    np.random.seed(self.config.seed)
    torch.manual_seed(self.config.seed)
    print('Original cat dist is {}'.format(cat_prob))

    # Priori configurations.
    z = torch.FloatTensor(self.config.batch_size, self.config.num_noise_dim).to(self.config.device)
    dis_c = torch.FloatTensor(self.config.batch_size, self.config.num_class).to(self.config.device)
    con_c = torch.FloatTensor(self.config.batch_size, self.config.num_con_c).to(self.config.device)
    real_label = 1
    fake_label = 0

    # Used to log numeric data.
    Glosses = []
    Dlosses = []
    EntQC_given_X = []
    MSEs = []
    generated_images = []

    criterionD = nn.BCELoss().to(self.config.device)
    criterionQ_dis = nn.CrossEntropyLoss().to(self.config.device)
    criterionQ_con = LogGaussian()
    # criterionQ_con = nn.MSELoss().to(self.config.device)
    # criterionQsemi = nn.CrossEntropyLoss().to(self.config.device)

    optimD = optim.Adam([{'params': self.FD.parameters()}, {'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params': self.G.parameters()}, {'params': self.Q.parameters()}, {'params': self.D.parameters()}], lr=0.001, betas=(0.5, 0.99))
    # optimizer = optim.Adam([{'params': self.G.parameters()}, 
    #                         {'params': self.FD.parameters()},
    #                         {'params': self.D.parameters()},
    #                         {'params': self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

    dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=1)
    tot_iters = len(dataloader)

    # Fixed random variables.
    fixz, cdis, c1, c2, c0 = self._fix_noise()
    fixed_noise_0 = np.hstack([fixz, cdis, c0])
    fixed_noise_1 = np.hstack([fixz, cdis, c1])
    fixed_noise_2 = np.hstack([fixz, cdis, c2])
    # NOTE: dtype should exactly match the network weight's type!
    fixed_noise_0 = torch.as_tensor(fixed_noise_0, dtype=torch.float32).view(100, -1, 1, 1).to(self.config.device)
    fixed_noise_1 = torch.as_tensor(fixed_noise_1, dtype=torch.float32).view(100, -1, 1, 1).to(self.config.device)
    fixed_noise_2 = torch.as_tensor(fixed_noise_2, dtype=torch.float32).view(100, -1, 1, 1).to(self.config.device)

    # Training...
    print('-'*25)
    print('Starting Training Loop...\n')
    print('Epochs: {}\nDataset: {}\nBatch size: {}\nLength of Dataloder: {}'
          .format(self.config.num_epoch, self.config.dataset, 
                  self.config.batch_size, len(dataloader)))
    print('-'*25)

    t0 = ETimer() # train timer
    t1 = ETimer() # epoch timer

    for epoch in range(self.config.num_epoch):
      t1.reset()
      for num_iter, (images, y) in enumerate(dataloader):
        ########################
        # Train discriminator.
        ########################
        ## real part
        optimD.zero_grad()

        bs = images.size(0) # alias for batch_size
        if (bs != self.config.batch_size):
          break
        z.resize_(bs, self.config.num_noise_dim)
        dis_c.resize_(bs, self.config.num_class)
        con_c.resize_(bs, self.config.num_con_c)

        images = images.to(self.config.device)
        images.requires_grad = True

        # Guided by https://github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        # This may be helpful for generator convergence.
        if self.config.instance_noise:
          instance_noise = torch.zeros_like(images)
          std = -0.1 / tot_iters * num_iter + 0.1
          instance_noise.normal_(0, std)
          images = images + instance_noise

        dbody_out_real = self.FD(images)
        _, probs_real = self.D(dbody_out_real)
        labels = torch.full_like(probs_real, real_label, device=self.config.device)
        loss_real = criterionD(probs_real, labels)
        loss_real.backward()

        ## fake part
        noise, idx = self._sample(z, dis_c, con_c, cat_prob, bs)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          fake_image = fake_image + instance_noise

        dbody_out_fake = self.FD(fake_image.detach())
        _, probs_fake = self.D(dbody_out_fake)
        labels.fill_(fake_label)
        loss_fake = criterionD(probs_fake, labels)
        loss_fake.backward()

        ## update
        D_loss = loss_real + loss_fake
        optimD.step()

        ############################
        # Train generator.
        ############################
        optimG.zero_grad()

        # NOTE: why bother to get a new output of FD? cause FD is updated by optimizing Discriminator.
        dbody_out = self.FD(fake_image)
        ut, probs_fake = self.D(dbody_out)
        labels.fill_(real_label)
        reconstruct_loss = criterionD(probs_fake, labels)

        q_logits, q_mu, q_var = self.Q(dbody_out)
        
        # # Entropy loss.
        # softmax = nn.Softmax(dim=1)
        # qc_given_x = softmax(q_logits)
        # ent_qc_given_x = torch.distributions.Categorical(probs=qc_given_x).entropy().mean()
        # EntQC_given_X.append(ent_qc_given_x.detach().item())
        # ent_loss = ent_qc_given_x * 0.4

        vt = q_logits[range(self.config.batch_size), idx.tolist()]
        new_loss = -(ut + vt).mean()
        # targets = torch.LongTensor(idx).to(self.config.device)
        # dis_loss = criterionQ_dis(q_logits, targets) * 0.8
        con_loss = criterionQ_con(con_c, q_mu, q_var) * 0.2 # weight

        ## update parameters
        G_loss = reconstruct_loss + new_loss + con_loss
        G_loss.backward()
        optimG.step()
        
        if (num_iter+1) % 100 == 0:
          print('Epoch: ({:3.0f}/{:3.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, len(dataloader), 
          D_loss.cpu().detach().numpy(), G_loss.cpu().detach().numpy())
          )

        Dlosses.append(D_loss.cpu().detach().item())
        Glosses.append(G_loss.cpu().detach().item())

      # Report epoch training time.
      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))

      if (epoch+1) % 2 == 0:
        img = self._save_image(fixed_noise_1, 'c0-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)

        # TODO: This is meaningless since category code must map to the correct ground truth label first.
        # rmse = np.linalg.norm(cat_prob - true_dist)
        # print('Current cat dist: {}\nRMSE: {:.4f}'.format(cat_prob, rmse))
      
      # # Saving checkpoint.
      # if (epoch+1) % self.config.save_epoch == 0:
      #   self._save_checkpoint('model-epoch-{}.pt'.format(epoch+1))

    # Training finished.
    training_time = t0.elapsed()
    print('-'*50)
    print('Traninig finished.\nTotal training time: %.2fm' % (training_time / 60))
    print('-'*50)
    generate_animation(self._savepath, generated_images)

    # Manipulating continuous latent codes.
    self._save_image(fixed_noise_1, 'c1-final.png')
    self._save_image(fixed_noise_2, 'c2-final.png')

    # Save the final model and losses.
    self._save_checkpoint('model-final.pt')
    np.savez(os.path.join(self._savepath, 'loss.npz'), Gloss=Glosses, Dloss=Dlosses, EntQ=EntQC_given_X, MSE=MSEs)
    return Glosses, Dlosses, EntQC_given_X, MSEs 
