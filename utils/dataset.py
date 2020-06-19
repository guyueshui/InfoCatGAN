import torch, numpy as np, copy
import torchvision.datasets as dsets
from torchvision import transforms

def DrawLabeled(dataset, num_labeled, report=False):
    """
    Cut a dataset to num_labeled samples, i.e., the result dataset contains
    only num_labeled data samples with corresponding targets.
    """
    assert num_labeled <= len(dataset), "too many samples to draw"
    
    dset = copy.deepcopy(dataset)
    if isinstance(dset, dsets.SVHN):
        num_class = 10
        dset.targets = dset.labels
    else:
        num_class = len(dset.class_to_idx)
    num_data_per_class = num_labeled // num_class
    
    counter = [num_data_per_class for i in range(num_class)]
    idx_to_draw = []
    for i, label in enumerate(dset.targets):
        if counter[label] > 0:
            idx_to_draw.append(i)
            counter[label] -= 1
    mask = np.zeros(len(dset), dtype=bool)
    mask[idx_to_draw] = True
    dset.data = dset.data[mask]
    dset.targets = np.asarray(dset.targets)[mask].tolist()

    if report:
        _, counts = np.unique(dset.targets, return_counts=True)
        print("-- Totally draw %d labeled samples" % len(dset.targets))
        print("-- The labeled dist is", counts / np.sum(counts))
    
    return dset
        

            

def GetData(dbname, data_root, train=True):
    "Get dataset from data root."
    
    if dbname == 'MNIST':
        trans = transforms.ToTensor()
        dataset = dsets.MNIST(data_root, train=train, transform=trans, download=True)

    elif dbname == 'SVHN':
        trans = transforms.Compose([
            transforms.Resize(37),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])
        split = 'train' if train else 'test'
        dataset = dsets.SVHN(data_root, split=split, transform=trans, download=True)
    
    elif dbname == 'CIFAR10':
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = dsets.CIFAR10(data_root, train=train, transform=trans, download=True)
    
    else:
        raise NotImplementedError

    return dataset