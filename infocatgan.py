import torch, numpy as np 
import os, time
import torchvision.utils as vutils
import torch.optim as optim
from torch.utils.data import DataLoader

from config import GetConfig
import utils

"""
load general configs, custom config should go here, once ARGS
passed to the gan object, it can not be changed
"""
ARGS = GetConfig()
if ARGS.dbname == 'MNIST':
    ARGS.nlabeled = 100
elif ARGS.dbname == 'CIFAR10':
    ARGS.nlabeled = 4000
elif ARGS.dbname == 'SVHN':
    ARGS.nlabeled = 4000

ARGS.num_epoch = 1000
if ARGS.seed == -1:
    ARGS.seed = int(time.time())

np.random.seed(ARGS.seed)
np.set_printoptions(precision=4)
torch.manual_seed(ARGS.seed)
torch.set_default_tensor_type(torch.FloatTensor)
#torch.cuda.empty_cache()


class InfoCatGAN(utils.BaseModel):
    def __init__(self, config):
        super(InfoCatGAN, self).__init__(config)
        self.nclass = 10
        self.z_dim = 128
        self.train_set = utils.GetData(config.dbname, config.data_root, True)
        self.test_set = utils.GetData(config.dbname, config.data_root, False)
        print("-- dataset info")
        print(self.train_set)
        print(self.test_set)
        self.log2file(self.train_set.__repr__())
        self.log2file(self.test_set.__repr__())

        self.log = {}
        self.log['d_loss'] = []
        self.log['g_loss'] = []
        self.log['info_loss'] = []
        self.log['fid'] = []

        self.modules = self.build_model()

    def Train(self, num_labeled=0):
        if num_labeled == 0:
            self.train()
        else:
            self.semi_train(num_labeled)

    def train(self):
        pass

    def semi_train(self, num_labeled):
        dv = self.device
        lr = 2e-4
        lr_anneal_factor = 0.995
        lr_anneal_epoch = 300 - 200
        lr_anneal_every_epoch = 1
        alpha_decay = 1e-4
        alpha_ent = 0.3
        alpha_avg = 1e-3
        eval_epoch = 5
        vis_epoch = 5
        epoch_g = 3   # after this epoch, some of generated samples will be used as real

        tot_timer = utils.ETimer()
        epoch_timer = utils.ETimer()

        testloader = DataLoader(self.test_set, batch_size=100, num_workers=4)
        fix_z = torch.randn(100, self.z_dim, device=dv)
        fix_code = torch.as_tensor(
            np.arange(self.nclass).repeat(10), device=dv
        ).long()
        fix_code_onehot = utils.Onehot(fix_code, self.nclass)
        fix_noise = torch.cat([fix_z, fix_code_onehot], dim=1)
        
        """
        Pretrain D
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
                
            acc = self.evaluate(testloader)
            line = "-- Pretrained acc at epoch %d: %.5f" % (epoch, acc)
            print(line)
            self.log2file(line)
        
        """
        train TMD
        """
        batch_size_l = min(num_labeled, 100)
        batch_size_u = 100
        supervised_prob = 0.99
        ce = torch.nn.CrossEntropyLoss().to(dv)
        alpha_info = 0.9

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
            dl = np.zeros(6)
            gl = np.zeros(4)
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
                d_cost_ent = utils.Entropy(d_out_u)
                # d_cost_ment = utils.CategoricalCrossentropyOfMean(d_out_u)
                d_cost_ment = utils.MarginalEntropy(d_out_u)

                if is_labeled_batch:
                    d_out_l = self.D(lx)
                    d_cost_bind = ce(d_out_l, ly)
                else:
                    d_cost_bind = torch.tensor(0.0)
                
                z = torch.randn(batch_size_u, self.z_dim, device=dv)
                code = torch.randint(self.nclass, size=(batch_size_u,), device=dv)
                code_onehot = utils.Onehot(code, self.nclass)
                noise = torch.cat([z, code_onehot], dim=1)
                fx = self.G(noise)

                d_out_f = self.D(fx.detach())
                d_cost_fake = utils.Entropy(d_out_f)
                #if num_iter % 200 == 0:
                #    print(d_out_f)
                #    print(d_cost_fake)
                #    print(d_cost_ment)
                # d_cost_fake.backward(torch.FloatTensor([-1]).to(dv))
                
                # use some of generated as real
                if epoch >= epoch_g:
                    g_z = torch.randn(batch_size_u, self.z_dim, device=dv)
                    g_code = torch.randint(self.nclass, size=(batch_size_u,), device=dv)
                    g_code_onehot = utils.Onehot(code, self.nclass)
                    g_noise = torch.cat([g_z, g_code_onehot], dim=1)
                    g_fx = self.G(g_noise)
                    d_out_g = self.D(g_fx.detach())
                    d_cost_bind_g = ce(d_out_g, g_code)
                else:
                    d_cost_bind_g = torch.tensor(0.0)
                
                d_cost = d_cost_ent - d_cost_ment - d_cost_fake + 1.1*d_cost_bind + .5*d_cost_bind_g
                d_cost_list = [d_cost, d_cost_ent, d_cost_ment, d_cost_fake, d_cost_bind, d_cost_bind_g]
                d_cost_list = [e.detach().cpu().item() for e in d_cost_list]
                for j in range(len(d_cost_list)):
                    dl[j] += d_cost_list[j]
                self.log['d_loss'].append(d_cost_list[0])
                d_cost.backward()
                d_optim.step()

                # Update G.
                g_optim.zero_grad()
                d_out_f = self.D(fx)
                g_cost_ent = utils.Entropy(d_out_f)
                # g_cost_ment = utils.CategoricalCrossentropyOfMean(d_out_f)
                g_cost_ment = utils.MarginalEntropy(d_out_f)
                g_cost_info = ce(d_out_f, code)
                g_cost = 1.11*g_cost_ent - g_cost_ment + alpha_info*g_cost_info
                g_cost_list = [g_cost, g_cost_ent, g_cost_ment, g_cost_info]
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
                supervised_prob = max(supervised_prob - 0.1, 0.09)
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
                    image_gen = self.G(fix_noise)
                    vutils.save_image(image_gen, self.save_dir + '/fake-epoch-{}.png'.format(epoch), nrow=10)
            
            if epoch % eval_epoch == 0:
                acc = self.evaluate(testloader)
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
        nchannel = self.train_set[0][0].size(0)
        import models.mnist as nets

        self.G = nets.G(self.z_dim + self.nclass, nchannel)
        self.D = nets.D(nchannel, self.nclass)
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
                pred = self.D(tx).argmax(axis=1).view(-1)
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
    gan = InfoCatGAN(ARGS)
    gan.load_model('/home/yychi/Documents/workspace/now_work/Torch-InfoGAN/results/MNIST/InfoCatGAN.nlabeled100.seed1.0619-13:29/model-epoch-200.pt', *gan.modules)
    gan.Train(ARGS.nlabeled)