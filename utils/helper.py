import torch, os, logging
import torch.nn as nn
import numpy as np
import time as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.use('AGG')

class BaseModel(object):
    "Base model that wrap some utilities for all implemented models."
    def __init__(self, config):
        self.config = config

        if config.gpu == 0:  # GPU selection.
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        elif config.gpu == 1:
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        elif config.gpu == -1:
            self.device = torch.device('cpu')
        else:
            raise IndexError('Invalid GPU index')

        save_dir = os.path.join('results', config.dbname, self.__class__.__name__)
        save_dir += '/nlabeled' + str(config.nlabeled)
        save_dir += '.seed' + str(config.seed)
        save_dir += '.' + config.tag
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir

        logging.basicConfig(filename=save_dir + '/run' + t.strftime("%m%d-%H:%M") + '.log', 
                            filemode='w', format="%(message)s",
                            level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        # Logging configurations.
        self.logger.info(">>> Experiment configurations")
        for item in config.__dict__:
            line = item + ": {}".format(config.__dict__[item])
            self.logger.info(line)
        self.logger.info("<<< -------------------------\n")
        
    
    def log2file(self, msg):
        self.logger.info(msg)

    def save_model(self, path, idx=None, *models):
        dic = {}
        for m in models:
            dic[m.__class__.__name__] = m.state_dict()
        fname = os.path.join(path, 'model-epoch-{}.pt'.format(idx))
        torch.save(dic, fname)
        print("-- model saved as ", fname)
    
    def load_model(self, fname, *models):
        params = torch.load(fname)
        for m in models:
            m.load_state_dict(params[m.__class__.__name__])
        print("-- load model from ", fname)
    
    def raw_classify(self, x):
        raise NotImplementedError


def Onehot(x, nclass :int):
    onehot = F.one_hot(x.long(), nclass)
    # onehot = onehot.type(x.dtype)
    # or
    # onehot.type(x.type())
    return onehot.float()

def WeightInit(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)

class ETimer:
    'A easy-to-use timer.'
 
    def __init__(self):
        self._start = t.time()

    def reset(self):
        self._start = t.time()

    def elapsed(self):  # In seconds.
        return t.time() - self._start

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    see: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/26

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

def plot_loss(log: dict, path: str):
    plt.style.use('ggplot')

    # loss
    # plt.figure(figsize=(10, 5))
    plt.title('GAN Loss')
    plt.plot(log['g_loss'], label='G', linewidth=1)
    plt.plot(log['d_loss'], label='D', linewidth=1)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path + '/gan_loss.png')
    plt.close('all')
    plt.plot(log['acc'], linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(path + '/acc.png')
    plt.close('all')