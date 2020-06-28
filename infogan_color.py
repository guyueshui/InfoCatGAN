import torch, numpy as np 
import os, time
import torchvision.utils as vutils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import GetConfig
import utils

g_lr = 1e-3
d_lr = 2e-4
lr_anneal_factor = 0.995
lr_anneal_epoch = 9999
lr_anneal_every_epoch = 1
alpha_decay = 1e-4
alpha_ent = 0.3
alpha_avg = 1e-3
epoch_g = 150   # after this epoch, some of generated samples will be used as real
vis_epoch = 2
eval_epoch = 1

tot_timer = utils.ETimer()
epoch_timer = utils.ETimer()

class InfoGAN(utils.BaseModel):
    def __init__(self, config):
        super(InfoGAN, self).__init__(config)
        self.nclass = 10
        self.z_dim = 124
        self.n_dis_c = 4
        self.n_con_c = 4
        self.train_set = utils.GetData(config.dbname, config.data_root, True)
        self.test_set = utils.GetData(config.dbname, config.data_root, False)
        self.log2file(">>> dataset info")
        self.log2file(self.train_set.__repr__())
        self.log2file(self.test_set.__repr__())
        self.log2file("----------------\n")

        # common stuff
        self.log = {}
        self.log['d_loss'] = []
        self.log['g_loss'] = []
        self.log['info_loss'] = []
        self.log['fid'] = []
        self.log['acc'] = []

        self.testloader = DataLoader(self.test_set, batch_size=100, num_workers=4)
        fix_z = torch.randn(100, self.z_dim, device=self.device)
        fix_con_c = torch.FloatTensor(100, self.n_con_c).to(self.device)
        fix_con_c.uniform_(-1, 1)

        fix_code = torch.as_tensor(
            np.arange(self.nclass).repeat(10), device=self.device
        ).long()
        fix_code_onehot = utils.Onehot(fix_code, self.nclass)
        self.fix_noise = torch.cat([fix_z] + [fix_code_onehot]*self.n_dis_c + [fix_con_c], dim=1)

        self.modules = self.build_model()

    def Train(self, num_labeled=0):
        if num_labeled == 0:
            self.train()
        else:
            self.semi_train(num_labeled)

    def train(self):
        raise NotImplementedError

    def semi_train(self, num_labeled):
        dv = self.device
        bs = 128

        z = torch.FloatTensor(bs, self.z_dim).to(dv)
        dis_c = torch.FloatTensor(bs, self.n_dis_c*self.nclass).to(dv)
        con_c = torch.FloatTensor(bs, self.n_con_c).to(dv)

        bce = nn.BCELoss().to(dv)
        ce = nn.CrossEntropyLoss().to(dv)
        mse = nn.MSELoss().to(dv)

        d_param_group = [
            {'params': self.FD.parameters()},
            {'params': self.D.parameters()},
        ]
        g_param_group = [
            {'params': self.G.parameters()},
            {'params': self.Q.parameters()},
        ]

        """
        Pretrain D
        """
        pre_epoch = 0 if num_labeled <= 100 else 0
        pre_batch_size_l = min(num_labeled, 128)
        pre_batch_size_u = 512
        pre_lr = 3e-4

        pre_d_optim = optim.Adam(
            [
                {'params':self.FD.parameters()},
                {'params':self.Q.parameters()},
            ],
            lr=pre_lr, weight_decay=alpha_decay
        )
        labeled_set = utils.DrawLabeled(self.train_set, num_labeled, True)
        for epoch in range(1, 1 + pre_epoch):
            uloader = DataLoader(self.train_set, batch_size=pre_batch_size_u, shuffle=True, num_workers=4, drop_last=True)
            lloader = DataLoader(labeled_set, batch_size=pre_batch_size_l, shuffle=True, drop_last=True)
            liter = iter(lloader)
            for ux, _ in uloader:
                try:
                    lbatch = next(liter)
                except StopIteration:
                    pass
                finally:
                    liter = iter(lloader)
                    lbatch = next(liter)
                ux = ux.to(dv)
                lx, ly = [e.to(dv) for e in lbatch]

                pre_d_optim.zero_grad()
                q_out_u, _, _ = self.Q(self.FD(ux))
                q_out_l, _, _ = self.Q(self.FD(lx))
                pre_d_cost_ent = utils.Entropy(F.softmax(q_out_u, dim=1))
                pre_d_cost_ment = utils.MarginalEntropy(F.softmax(q_out_u, dim=1))
                pre_d_cost_bind = ce(q_out_l, ly)
                pre_d_cost = pre_d_cost_ent - pre_d_cost_ment + 1.1*pre_d_cost_bind
                pre_d_cost.backward()
                pre_d_optim.step()
            
            acc = self.evaluate(self.testloader)
            line = "-- Pretrained acc at epoch %d: %.5f" % (epoch, acc)
            print(line)
            self.log2file(line)
        
        """
        train TMD
        """
        batch_size_l = min(num_labeled, bs)
        batch_size_u = bs
        supervised_prob = 0.99
        alpha_info = 0.9

        # Training...
        print('-'*25)
        print('Starting Training Loop...\n')
        print('Epochs: {}\nDataset: {}'.format(self.config.num_epoch, self.config.dbname))
        print('-'*25)
        tot_timer.reset()
        for epoch in range(1, 1 + self.config.num_epoch):
            epoch_timer.reset()
            g_optim = optim.Adam(g_param_group, lr=g_lr, betas=(0.5, 0.999))
            d_optim = optim.Adam(d_param_group, lr=d_lr, betas=(0.5, 0.999))
            q_optim = optim.Adam(self.Q.parameters(), lr=g_lr, betas=(0.5, 0.999), weight_decay=alpha_decay)

            uloader = DataLoader(self.train_set, batch_size=batch_size_u, shuffle=True, num_workers=4, drop_last=True)
            lloader = DataLoader(labeled_set, batch_size=batch_size_l, shuffle=True, num_workers=2, drop_last=True)
            liter = iter(lloader)
            dl = np.zeros(3)
            gl = np.zeros(6)
            for num_iter, (ux, _) in enumerate(uloader, 1):
                # Prepare data.
                ux = ux.to(dv)
                is_labeled_batch = (torch.bernoulli(torch.tensor(supervised_prob)) == 1)
                if is_labeled_batch:
                    try:
                        lbatch = next(liter)
                    except StopIteration:
                        liter = iter(lloader)
                        lbatch = next(liter)

                    lx, ly = [e.to(dv) for e in lbatch]

                # Update D.
                d_optim.zero_grad()
                d_body_out_u = self.FD(ux)
                d_out_u = self.D(d_body_out_u)
                labels = torch.full_like(d_out_u, 1, device=dv)
                d_cost_real = bce(d_out_u, labels)

                if is_labeled_batch:
                    noise, idx = self.generate_noise(z, dis_c, con_c, ly)
                else:
                    noise, idx = self.generate_noise(z, dis_c, con_c)
                idx = torch.LongTensor(idx).to(dv)

                fx = self.G(noise)
                d_body_out_f = self.FD(fx.detach())

                # update Q to bind real label if cur_batch is_labeled_batch
                if is_labeled_batch:
                    q_optim.zero_grad()
                    disc_logits_real, _, _ = self.Q(self.FD(lx))
                    qsuper_loss_real = ce(disc_logits_real[:,:self.nclass], ly)
                    qsuper_loss_real.backward()
                    q_optim.step()
                elif epoch >= epoch_g:
                    q_optim.zero_grad()
                    disc_logits_fake_as_real, _, _ = self.Q(d_body_out_f.detach())
                    qsuper_loss_fake_as_real = ce(disc_logits_fake_as_real[:,:self.nclass], idx[0])*0.1
                    qsuper_loss_fake_as_real.backward()
                    q_optim.step()

                d_out_f = self.D(d_body_out_f)
                labels = torch.full_like(d_out_f, 0, device=dv)
                d_cost_fake = bce(d_out_f, labels)

                d_cost = d_cost_real + d_cost_fake
                d_cost_list = [d_cost, d_cost_real, d_cost_fake]
                d_cost_list = [e.detach().cpu().item() for e in d_cost_list]
                for j in range(len(d_cost_list)):
                    dl[j] += d_cost_list[j]
                self.log['d_loss'].append(d_cost_list[0])
                d_cost.backward()
                d_optim.step()

                # Update G.
                g_optim.zero_grad()
                d_body_out_f = self.FD(fx)
                d_out_f = self.D(d_body_out_f)
                labels = torch.full_like(d_out_f, 1, device=dv)
                g_cost_dis = bce(d_out_f, labels)

                q_logits, q_mu, q_var = self.Q(d_body_out_f)
                if is_labeled_batch:
                    xent = utils.CategoricalCrossentropy(
                        F.softmax(q_logits[:,:self.nclass], dim=1),
                        F.softmax(disc_logits_real.detach()[:,:self.nclass], dim=1)
                    )
                else:
                    xent = torch.tensor(0)

                q_cost_dis = 0
                for j in range(self.n_dis_c):
                    q_cost_dis += ce(q_logits[:, j*self.nclass : (j+1)*self.nclass], idx[j])
                q_cost_con = utils.LogGaussian(con_c, q_mu, q_var)

                # feature matching loss
                fmatch_loss = mse(d_body_out_f, d_body_out_u.detach())

                g_cost = g_cost_dis + q_cost_dis + .5*q_cost_con + .9*fmatch_loss + .9*xent
                g_cost_list = [g_cost, g_cost_dis, q_cost_dis, q_cost_con, fmatch_loss, xent]
                g_cost_list = [e.detach().cpu().item() for e in g_cost_list]
                for j in range(len(g_cost_list)):
                    gl[j] += g_cost_list[j]
                self.log['g_loss'].append(g_cost_list[0])
                g_cost.backward()
                g_optim.step()

            # end of epoch
            dl /= len(uloader)
            gl /= len(uloader)

            if supervised_prob > 0.1:
                supervised_prob = max(supervised_prob - 0.1, 0.1)
            if (epoch >= lr_anneal_epoch) and (epoch % lr_anneal_every_epoch == 0):
                lr *= lr_anneal_factor
            line = "Epoch=%d Time=%.2f LR=%.5f\n" % (epoch, epoch_timer.elapsed(), g_lr) +\
                "Dlosses: " + str(dl) + "\nGlosses: " + str(gl) + '\n'
            print(line)
            self.log2file(line)

            if epoch % 100 == 0:
                self.save_model(self.save_dir, epoch, *self.modules)

            if epoch % vis_epoch == 0:
                with torch.no_grad():
                    image_gen = self.G(self.fix_noise)
                    vutils.save_image(image_gen, self.save_dir + '/fake-epoch-{}.png'.format(epoch), nrow=10, normalize=True)
            
            if epoch % eval_epoch == 0:
                acc = self.evaluate(self.testloader)
                self.log['acc'].append(acc)
                line = "AccEval=%.5f\n" % acc
                print(line)
                self.log2file(line)

        # Training finished.
        print('-'*50)
        print('Traninig finished.\nTotal training time: %.2fm' % (tot_timer.elapsed() / 60))
        self.save_model(self.save_dir, self.config.num_epoch, *self.modules)
        print('-'*50)
        utils.plot_loss(self.log, self.save_dir)



    def build_model(self):
        c, h, w = self.train_set[0][0].size()
        import models.svhn as nets
        noise_dim = self.z_dim + self.nclass*self.n_dis_c + self.n_con_c
        latent_dim = 256

        self.G = nets.Generator(noise_dim, c)
        self.FD = nets.Dbody(c, latent_dim)
        self.D = nets.Dhead(latent_dim, 1)
        self.Q = nets.Qhead(latent_dim, self.n_dis_c, self.n_con_c)

        networks = [self.G, self.FD, self.D, self.Q]
        for i in networks:
            i.apply(utils.WeightInit)
            i.to(self.device)
        return networks


    def evaluate(self, testloader, need_map=False):
        preds = []
        targets = []
        for tx, ty in testloader:
            tx = tx.to(self.device)
            with torch.no_grad():
                pred, _, _ = self.Q(self.FD(tx))
                pred = pred.argmax(axis=1).view(-1)
                preds.append(pred)
                targets.append(ty)
        preds = torch.cat(preds, axis=-1)
        targets = torch.cat(targets, axis=-1)
        if need_map:
            map_to_real = utils.CategoryMatching(preds, targets, self.nclass)
        else:
            map_to_real = None
        acc = utils.CategoricalAccuracy(preds, targets, map_to_real)
        return acc
    
    def generate_noise(self, z, dis_c, con_c, y=None):
        bs = z.size(0)
        z.normal_(0, 1)
        con_c.uniform_(-1, 1)

        discode = torch.zeros(bs, self.n_dis_c, self.nclass)
        idxes = np.zeros((self.n_dis_c, bs))
        for i in range(self.n_dis_c):
            if i == 0 and y is not None:
                idxes[i] = y.cpu().numpy()
            else:
                idxes[i] = np.random.randint(self.nclass, size=bs)
            discode[range(bs), i, idxes[i]] = 1.0
        discode = discode.view_as(dis_c)
        dis_c.copy_(discode)
        noise = torch.cat([z, dis_c, con_c], dim=1)
        return noise, idxes
            


        
if __name__ == '__main__':
    """
    load general configs, custom config should go here, once ARGS
    passed to the gan object, it can not be changed
    """
    ARGS = GetConfig()
    assert ARGS.dbname in ['SVHN', 'CIFAR10']
    #ARGS.nlabeled = 132
    
    ARGS.num_epoch = 300
    if ARGS.seed == -1:
        ARGS.seed = int(time.time())
    
    np.random.seed(ARGS.seed)
    np.set_printoptions(precision=4)
    torch.manual_seed(ARGS.seed)
    torch.set_default_tensor_type(torch.FloatTensor)

    gan = InfoGAN(ARGS)
    #gan.load_model('results/FashionMNIST/InfoCatGAN/nlabeled0.seed1.lamb_info.9/model-epoch-300.pt', *gan.modules)
    gan.Train(ARGS.nlabeled)

    if ARGS.fid:
        dl = DataLoader(gan.train_set, 100, num_workers=4)
        gen_imgs = []
        for _, _ in dl:
            noise, _ = gan.generate_noise(100)
            with torch.no_grad():
                img_tensor = gan.G(noise)
                img_list = [i for i in img_tensor]
                gen_imgs.extend(img_list)
        fid_value = utils.ComputeFID(gen_imgs, gan.train_set, gan.device)
        print("-- FID score %.4f" % fid_value)
