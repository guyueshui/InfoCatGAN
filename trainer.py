import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from utils import BlahutArimoto, Noiser, LogGaussian, ETimer, generate_animation, \
  CustomDataset, Entropy, MarginalEntropy

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

  def ss_train(self, cat_prob, supervised_ratio):
    torch.set_default_tensor_type(torch.FloatTensor)
    np.set_printoptions(precision=4)
    torch.manual_seed(self.config.seed)
    np.random.seed(self.config.seed)

    # Used to log numeric data.
    Glosses = []
    Dlosses = []
    EntQC_given_X = []
    MSEs = []
    generated_images = []

    bs = self.config.batch_size
    dv = self.config.device

    z = torch.FloatTensor(bs, self.config.num_noise_dim).to(dv)
    disc_c = torch.FloatTensor(bs, self.config.num_class).to(dv)
    cont_c = torch.FloatTensor(bs, self.config.num_con_c).to(dv)
    real_label = 1
    fake_label = 0

    criterionD = nn.BCELoss().to(dv)
    criterionQ_dis = nn.CrossEntropyLoss().to(dv)
    criterionQ_con = LogGaussian()
    
    optimD = optim.Adam([{'params': self.FD.parameters()}, {'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params': self.G.parameters()}, {'params': self.Q.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    #! I have confirmed this is a good option. optimG = optim.Adam([{'params': self.G.parameters()}], lr=0.001, betas=(0.5, 0.99))
    # optimQ = optim.Adam(self.Q.parameters(), lr=0.0005, betas=(0.5, 0.99))

    dset = CustomDataset(self.dataset, supervised_ratio)
    dset.report()
    labeled_loder = DataLoader(dset.labeled, batch_size=bs, shuffle=True, num_workers=1)
    labeled_iter = iter(labeled_loder)
    unlabeled_loder = DataLoader(dset.unlabeled, batch_size=bs, shuffle=True, num_workers=1)
    unlabeled_iter = iter(unlabeled_loder)

    # Fixed random variables.
    fixz, cdis, c1, c2, c0 = self._fix_noise()
    fixed_noise_0 = np.hstack([fixz, cdis, c0])
    fixed_noise_1 = np.hstack([fixz, cdis, c1])
    fixed_noise_2 = np.hstack([fixz, cdis, c2])
    # NOTE: dtype should exactly match the network weight's type!
    fixed_noise_0 = torch.as_tensor(fixed_noise_0, dtype=torch.float32).view(100, -1, 1, 1).to(dv)
    fixed_noise_1 = torch.as_tensor(fixed_noise_1, dtype=torch.float32).view(100, -1, 1, 1).to(dv)
    fixed_noise_2 = torch.as_tensor(fixed_noise_2, dtype=torch.float32).view(100, -1, 1, 1).to(dv)

    # Training...
    print('-'*25)
    print('Starting Training Loop...\n')
    print('Epochs: {}\nDataset: {}\nBatch size: {}\nLength of Dataloder: {}'
          .format(self.config.num_epoch, self.config.dataset, 
                  self.config.batch_size, 100))
    print('-'*25)

    t0 = ETimer() # train timer
    t1 = ETimer() # epoch timer
    supervised_prob = 0.99

    for epoch in range(self.config.num_epoch):
      t1.reset()
      num_labeled_batch = 0
      if supervised_prob >= 0.1:
        tot_iters = 100
      else:
        tot_iters = len(unlabeled_loder) + len(labeled_loder)
      for num_iter in range(tot_iters):
        #
        # Train discriminator.
        #
        optimD.zero_grad()
        # Real part.
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
          print(cur_batch)
          print(not cur_batch)

        if not cur_batch or cur_batch[0].size(0) != bs:
          if is_labeled_batch:
            labeled_iter = iter(labeled_loder)
            cur_batch = next(labeled_iter)
            num_labeled_batch += 1
          else:
            unlabeled_iter = iter(unlabeled_loder)
            cur_batch = next(unlabeled_iter)

        image, y = cur_batch
        image = image.to(dv)

        # Guided by https://github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        # This may be helpful for generator convergence.
        if self.config.instance_noise:
          instance_noise = torch.zeros_like(image)
          std = -0.1 / tot_iters * num_iter + 0.1
          instance_noise.normal_(0, std)
          image = image + instance_noise

        dbody_out_real = self.FD(image)
        probs_real = self.D(dbody_out_real)
        # Minimize entropy to make certain prediction of real sample.
        ent_real = Entropy(probs_real)
        # Maximize marginal entropy over all real samples to ensure equal usage.
        margin_ent_real = MarginalEntropy(probs_real)

        if is_labeled_batch:
          optimQ.zero_grad()
          disc_logits_real, _, _ = self.Q(dbody_out_real.detach())
          qsemi_loss_real = criterionQ_dis(disc_logits_real, y.to(dv)) * 1.5
          qsemi_loss_real.backward()
          optimQ.step()
        if is_labeled_batch:
          sup_loss = criterionQ_dis(q_logits, targets) * 1.1
        else:
          sup_loss = criterionQ_dis(q_logits, targets) * 1.0

        # Fake part.
        noise = torch.randn(bs, self.config.num_noise_dim, 1, 1).to(dv)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          # instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

        dbody_out_fake = self.FD(fake_image.detach())
        probs_fake = self.D(dbody_out_fake)
        # Maximize entropy to make uncertain prediction of fake sample.
        ent_fake = Entropy(probs_fake)

        ## update
        D_loss = ent_real - margin_ent_real - ent_fake
        D_loss.backward()
        optimD.step()
        
        #
        # Train generator.
        #
        optimG.zero_grad()
        # NOTE: why bother to get a new output of FD? cause FD is updated by optimizing Discriminator.
        noise = torch.randn(bs, self.config.num_noise_dim, 1, 1).to(dv)
        fake_image = self.G(noise)
        dbody_out = self.FD(fake_image)
        probs_fake = self.D(dbody_out)
        # Minimize entropy to fool D to make a certain prediction of fake sample.
        ent_fake = Entropy(probs_fake)
        # Maxmize marginal entropy to ensure equal usage.
        margin_ent_fake = MarginalEntropy(probs_fake)

        ## update parameters
        G_loss = ent_fake - margin_ent_fake
        G_loss.backward()
        optimG.step()
        
        if (num_iter+1) % 50 == 0:
          print('Epoch: ({:3.0f}/{:3.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, tot_iters, 
          D_loss.cpu().detach().numpy(), G_loss.cpu().detach().numpy())
          )

        Dlosses.append(D_loss.cpu().detach().item())
        Glosses.append(G_loss.cpu().detach().item())

      # Report epoch training time.
      epoch_time = t1.elapsed()
      print('Time taken for Epoch %d: %.2fs' % (epoch+1, epoch_time))
      print('labeled batch for Epoch %d: %d/%d' % (epoch+1, num_labeled_batch, tot_iters))
      if supervised_prob > 0.01:
        supervised_prob -= 0.1
      if supervised_prob < 0.01:
        supervised_prob = 0.01

      if (epoch+1) % 2 == 0:
        img = self._save_image(fixed_noise_1, 'c0-epoch-{}.png'.format(epoch+1))
        generated_images.append(img)

      if (epoch+1) % self.config.save_epoch == 0:
        self._save_checkpoint('model-epoch-{}.pt'.format(epoch+1))

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


  def train(self, cat_prob):
    torch.set_default_tensor_type(torch.FloatTensor)
    np.set_printoptions(precision=4)
    torch.manual_seed(self.config.seed)
    np.random.seed(self.config.seed)

    # Used to log numeric data.
    Glosses = []
    Dlosses = []
    EntQC_given_X = []
    MSEs = []
    generated_images = []

    bs = self.config.batch_size
    dv = self.config.device

    z = torch.FloatTensor(bs, self.config.num_noise_dim).to(dv)
    disc_c = torch.FloatTensor(bs, self.config.num_class).to(dv)
    cont_c = torch.FloatTensor(bs, self.config.num_con_c).to(dv)
    
    optimD = optim.Adam([{'params': self.FD.parameters()}, {'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    # optimG = optim.Adam([{'params': self.G.parameters()}, {'params': self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params': self.G.parameters()}], lr=0.001, betas=(0.5, 0.99)) # I have confirmed this is a good option.

    dataloader = DataLoader(self.dataset, batch_size=bs, shuffle=True, num_workers=1)
    tot_iters = len(dataloader)

    # Fixed random variables.
    fixz, cdis, c1, c2, c0 = self._fix_noise()
    # fixed_noise_0 = np.hstack([fixz, cdis, c0])
    # fixed_noise_1 = np.hstack([fixz, cdis, c1])
    # fixed_noise_2 = np.hstack([fixz, cdis, c2])
    fixed_noise_0 = fixz
    fixed_noise_1 = fixz
    fixed_noise_2 = fixz
    # NOTE: dtype should exactly match the network weight's type!
    fixed_noise_0 = torch.as_tensor(fixed_noise_0, dtype=torch.float32).view(100, -1, 1, 1).to(dv)
    fixed_noise_1 = torch.as_tensor(fixed_noise_1, dtype=torch.float32).view(100, -1, 1, 1).to(dv)
    fixed_noise_2 = torch.as_tensor(fixed_noise_2, dtype=torch.float32).view(100, -1, 1, 1).to(dv)

    # Training...
    print('-'*25)
    print('Starting Training Loop...\n')
    print('Epochs: {}\nDataset: {}\nBatch size: {}\nLength of Dataloder: {}'
          .format(self.config.num_epoch, self.config.dataset, 
                  self.config.batch_size, tot_iters))
    print('-'*25)

    t0 = ETimer() # train timer
    t1 = ETimer() # epoch timer

    for epoch in range(self.config.num_epoch):
      t1.reset()
      for num_iter, cur_batch in enumerate(dataloader):
        #
        # Train discriminator.
        #
        optimD.zero_grad()

        image, _ = cur_batch
        image = image.to(dv)
        if image.size(0) != bs:
          break # start next epoch

        # Guided by https://github.com/soumith/ganhacks#13-add-noise-to-inputs-decay-over-time
        # This may be helpful for generator convergence.
        if self.config.instance_noise:
          instance_noise = torch.zeros_like(image)
          std = -0.1 / tot_iters * num_iter + 0.1
          instance_noise.normal_(0, std)
          image = image + instance_noise

        dbody_out_real = self.FD(image)
        probs_real = self.D(dbody_out_real)
        print(probs_real.size())
        # Minimize entropy to make certain prediction of real sample.
        ent_real = Entropy(probs_real)
        # Maximize marginal entropy over real samples to ensure equal usage.
        margin_ent_real = MarginalEntropy(probs_real)

        # Fake part.
        noise = torch.randn(bs, self.config.num_noise_dim, 1, 1).to(dv)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          # instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

        dbody_out_fake = self.FD(fake_image.detach())
        probs_fake = self.D(dbody_out_fake)
        # Maximize entropy to make uncertain prediction of fake sample.
        ent_fake = Entropy(probs_fake)

        ## update
        D_loss = ent_real - margin_ent_real - ent_fake
        D_loss.backward()
        optimD.step()
        
        #
        # Train generator.
        #
        optimG.zero_grad()

        noise = torch.randn(bs, self.config.num_noise_dim, 1, 1).to(dv)
        fake_image = self.G(noise)

        ## Add instance noise if specified.
        if self.config.instance_noise:
          # instance_noise.normal_(0, std)  # Regenerate instance noise!
          fake_image = fake_image + instance_noise

        # NOTE: why bother to get a new output of FD? cause FD is updated by optimizing Discriminator.
        dbody_out = self.FD(fake_image)
        probs_fake = self.D(dbody_out)

        # Fool D to make it believe the fake is real.
        ent_fake = Entropy(probs_fake)
        # Ensure equal usage of fake samples.
        margin_ent_fake = MarginalEntropy(probs_fake)

        ## update parameters
        G_loss = ent_fake - margin_ent_fake
        G_loss.backward()
        optimG.step()
        
        if (num_iter+1) % 50 == 0:
          print('Epoch: ({:3.0f}/{:3.0f}), Iter: ({:3.0f}/{:3.0f}), Dloss: {:.4f}, Gloss: {:.4f}'
          .format(epoch+1, self.config.num_epoch, num_iter+1, tot_iters, 
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

      if (epoch+1) % self.config.save_epoch == 0:
        self._save_checkpoint('model-epoch-{}.pt'.format(epoch+1))

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
