import torch, numpy as np 
import os, time
import torchvision.utils as vutils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import GetConfig
import utils

lr = 1e-3
lr_anneal_factor = 0.995
lr_anneal_epoch = 350
lr_anneal_every_epoch = 1
alpha_decay = 9e-4
alpha_ent = 0.3
alpha_avg = 1e-3
epoch_g = 300   # after this epoch, some of generated samples will be used as real
vis_epoch = 2
eval_epoch = 1

tot_timer = utils.ETimer()
epoch_timer = utils.ETimer()

class CatGAN(utils.BaseModel):
    def __init__(self, config):
        super(CatGAN, self).__init__(config)
        self.nclass = 10
        self.z_dim = 128 + 15
        self.train_set = utils.GetData(config.dbname, config.data_root, train=True)
        self.test_set = utils.GetData(config.dbname, config.data_root, train=False)
        self.log2file(">>> dataset info")
        self.log2file(self.train_set.__repr__())
        self.log2file(self.test_set.__repr__())
        self.log2file("----------------\n")

        self.log = {}
        self.log['d_loss'] = []
        self.log['g_loss'] = []
        self.log['fid'] = []
        self.log['ent'] = []
        self.log['acc'] = []

        self.testloader = DataLoader(self.test_set, batch_size=100, num_workers=4)
        self.fix_noise = torch.randn(100, self.z_dim, device=self.device)

        self.modules = self.build_model()
    
    def Train(self, num_labeled=0):
        if num_labeled == 0:
            self.train()
        else:
            self.semi_train(num_labeled)
    
    def train(self):
        dv = self.device
        bce = nn.BCELoss().to(dv)

        """
        train TMD
        """
        batch_size_u = 100
        global lr

        # Training...
        print('-'*25)
        print('Starting Training Loop...\n')
        print('Epochs: {}\nDataset: {}'.format(self.config.num_epoch, self.config.dbname))
        print('-'*25)
        tot_timer.reset()
        for epoch in range(1, 1 + self.config.num_epoch):
            epoch_timer.reset()
            g_optim = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
            d_optim = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=alpha_decay)

            uloader = DataLoader(self.train_set, batch_size=batch_size_u, shuffle=True, num_workers=4, drop_last=True)
            dl = np.zeros(5)
            gl = np.zeros(3)
            for num_iter, (ux, _) in enumerate(uloader, 1):
                # Prepare data.
                ux = ux.to(dv)

                # Update D.
                d_optim.zero_grad()
                dis_out, d_out_u = self.D(ux)
                d_cost_real = utils.CategoricalCrossentropyUslSeparated(
                    F.softmax(d_out_u, dim=1), alpha_ent, alpha_avg
                )
                labels = torch.full_like(dis_out, 1, device=dv)
                d_cost_dis_real = bce(dis_out, labels)

                # prepare fake data
                noise = torch.randn(batch_size_u, self.z_dim, device=dv)
                fx = self.G(noise)
                dis_out_f, d_out_f = self.D(fx.detach())
                d_cost_fake = utils.Entropy(F.softmax(d_out_f, dim=1))
                labels = torch.full_like(dis_out_f, 0, device=dv)
                d_cost_dis_fake = bce(dis_out_f, labels)
                d_cost = d_cost_dis_real + d_cost_dis_fake + (d_cost_real - alpha_ent*d_cost_fake)
                
                d_cost_list = [d_cost, d_cost_dis_real, d_cost_dis_fake, d_cost_real, d_cost_fake]
                d_cost_list = [e.detach().cpu().item() for e in d_cost_list]
                self.log['d_loss'].append(d_cost_list[0])
                for j in range(len(d_cost_list)):
                    dl[j] += d_cost_list[j]
                d_cost.backward()
                d_optim.step()
                
                # Update G.
                g_optim.zero_grad()
                dis_out_fake, d_out_f_g = self.D(fx)
                g_cost_cat = utils.CategoricalCrossentropyUslSeparated(
                    F.softmax(d_out_f_g, dim=1), alpha_ent, alpha_avg
                )
                labels = torch.full_like(dis_out_fake, 1, device=dv)
                g_cost_dis = bce(dis_out_fake, labels)
                g_cost = g_cost_dis + g_cost_cat
                g_cost_list = [g_cost, g_cost_dis, g_cost_cat]
                g_cost_list = [e.detach().cpu().item() for e in g_cost_list]
                for j in range(len(g_cost_list)):
                    gl[j] += g_cost_list[j]
                self.log['g_loss'].append(g_cost_list[0])
                g_cost.backward()
                g_optim.step()
            # end of epoch
            dl /= len(uloader)
            gl /= len(uloader)

            if (epoch >= lr_anneal_epoch) and (epoch % lr_anneal_every_epoch == 0):
                lr *= lr_anneal_factor
            line = "Epoch=%d Time=%.2f LR=%.5f\n" % (epoch, epoch_timer.elapsed(), lr) +\
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
                acc = self.evaluate(self.testloader, True)
                self.log['acc'].append(acc)
                line = "AccEval=%.5f" % acc
                print(line)
                self.log2file(line)

        # Training finished.
        print('-'*50)
        print('Traninig finished.\nTotal training time: %.2fm' % (tot_timer.elapsed() / 60))
        self.save_model(self.save_dir, self.config.num_epoch, *self.modules)
        print('-'*50)
        utils.plot_loss(self.log, self.save_dir)



    def semi_train(self, num_labeled):
        dv = self.device

        """
        pretrain D
        """
        pre_nepoch = 0 if num_labeled > 100 else 0
        pre_batch_size_l = min(num_labeled, 100)
        pre_batch_size_u = 500
        pre_lr = 3e-4

        d_optim = optim.Adam(self.D.parameters(), lr=pre_lr, weight_decay=alpha_decay)
        labeled_set = utils.DrawLabeled(self.train_set, num_labeled)
        for epoch in range(1, 1 + pre_nepoch):
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

                d_optim.zero_grad()
                d_out_u = self.D(ux)
                d_out_l = self.D(lx)
                d_cost = utils.CategoricalCrossentropySslSeparated(d_out_l, ly, d_out_u)
                d_cost.backward()
                d_optim.step()
                
            acc = self.evaluate(self.testloader)
            line = "-- Pretrained acc at epoch %d: %.5f" % (epoch, acc)
            print(line)
            self.log2file(line)


        """
        train TMD
        """
        batch_size_l = min(num_labeled, 100)
        batch_size_u = 100
        supervised_prob = 0.99

        # Training...
        print('-'*25)
        print('Starting Training Loop...\n')
        print('Epochs: {}\nDataset: {}'.format(self.config.num_epoch, self.config.dbname))
        print('-'*25)
        tot_timer.reset()
        for epoch in range(1, 1 + self.config.num_epoch):
            epoch_timer.reset()
            g_optim = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
            d_optim = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=alpha_decay)

            uloader = DataLoader(self.train_set, batch_size=batch_size_u, shuffle=True, num_workers=4, drop_last=True)
            lloader = DataLoader(labeled_set, batch_size=batch_size_l, shuffle=True, drop_last=True)
            liter = iter(lloader)
            dl = np.zeros(3)
            gl = np.zeros(1)
            for num_iter, (ux, _) in enumerate(uloader, 1):
                # Prepare data.
                ux = ux.to(dv)
                is_labeled_batch = (torch.bernoulli(torch.tensor(supervised_prob)) == 1)
                if is_labeled_batch:
                    try:
                        lbatch = next(liter)
                    except StopIteration:
                        pass
                    finally:
                        liter = iter(lloader)
                        lbatch = next(liter)

                    lx, ly = [e.to(dv) for e in lbatch]

                # Update D.
                d_optim.zero_grad()
                d_out_u = self.D(ux)
                
                if is_labeled_batch:
                    d_out_l = self.D(lx)
                    d_cost_real = utils.CategoricalCrossentropySslSeparated(
                        d_out_l, ly, d_out_u, 1, alpha_ent, alpha_avg)
                else:
                    d_cost_real = utils.CategoricalCrossentropyUslSeparated(
                        d_out_u, alpha_ent, alpha_avg)
                
                noise = torch.randn(batch_size_u, self.z_dim, device=dv)
                fx = self.G(noise)
                d_out_f = self.D(fx.detach())
                d_cost_fake = utils.Entropy(d_out_f)

                d_cost = d_cost_real - alpha_ent*d_cost_fake
                d_cost_list = [d_cost, d_cost_real, d_cost_fake]
                d_cost_list = [e.detach().cpu().item() for e in d_cost_list]
                for j in range(len(d_cost_list)):
                    dl[j] += d_cost_list[j]
                self.log['d_loss'].append(d_cost_list[0])
                d_cost.backward()
                d_optim.step()

                # Update G.
                g_optim.zero_grad()
                d_out_f = self.D(fx)
                g_cost = utils.CategoricalCrossentropyUslSeparated(d_out_f, alpha_ent, alpha_avg)
                g_cost_list = [g_cost]
                g_cost_list = [e.detach().cpu().item() for e in g_cost_list]
                for j in range(len(g_cost_list)):
                    gl[j] += g_cost_list[j]
                self.log['g_loss'].append(g_cost_list[0])
                g_cost.backward()
                g_optim.step()

                #if num_iter % 200 == 0:
                #    line = 'Epoch: {}, Iter: {}, Dloss: {:.4f}, Gloss: {:.4f}'.format(
                #        epoch, num_iter, d_cost.cpu().detach().numpy(), g_cost.cpu().detach().numpy())
                #    print(line)
                #    self.log2file(line)
            # end of epoch
            dl /= len(uloader)
            gl /= len(uloader)

            if supervised_prob > 0.01:
                supervised_prob = max(supervised_prob - 0.05, 0.01)
            if (epoch >= lr_anneal_epoch) and (epoch % lr_anneal_every_epoch == 0):
                lr *= lr_anneal_factor
            line = "Epoch=%d Time=%.2f LR=%.5f\n" % (epoch, epoch_timer.elapsed(), lr) +\
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
        c,h,w = self.train_set[0][0].size()
        import models.cifar10 as nets
        self.G = nets.DCGAN_G(self.z_dim, (c,h,w))
        self.D = nets.DCGAN_CATFD((c,h,w), self.nclass)
        networks = [self.G, self.D]
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
                _, pred = self.D(tx)
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
        
        

if __name__ == '__main__':
    """
    load general configs, custom config should go here, once ARGS
    passed to the gan object, it can not be changed
    """
    ARGS = GetConfig()
    assert (ARGS.dbname in ['SVHN', 'CIFAR10'])
    
    ARGS.num_epoch = 500
    if ARGS.seed == -1:
        ARGS.seed = int(time.time())
    
    np.random.seed(ARGS.seed)
    np.set_printoptions(precision=4)
    torch.manual_seed(ARGS.seed)
    torch.set_default_tensor_type(torch.FloatTensor)

    gan = CatGAN(ARGS)
    #gan.load_model('results/FashionMNIST/CatGAN/nlabeled100.seed1.default/model-epoch-500.pt', *gan.modules)
    gan.Train(ARGS.nlabeled)

    if ARGS.fid:
        dl = DataLoader(gan.train_set, 100, num_workers=4)
        gen_imgs = []
        for _, _ in dl:
            noise = torch.randn(100, gan.z_dim, device=gan.device)
            with torch.no_grad():
                img_tensor = gan.G(noise)
                img_list = [i for i in img_tensor]
                gen_imgs.extend(img_list)
        fid_value = utils.ComputeFID(gen_imgs, gan.train_set, gan.device)
        print("-- FID score %.4f" % fid_value)

