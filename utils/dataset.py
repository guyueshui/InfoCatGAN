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
    
    data = []
    targets = []
    for j in range(num_class):
        data.append(dset.data[dset.targets == j][:num_data_per_class])
        targets.append(dset.targets[dset.targets == j][:num_data_per_class])
    dset.data = torch.cat(data, dim=0)
    dset.targets = torch.cat(targets, dim=0)

    if report:
        _, counts = np.unique(dset.targets, return_counts=True)
        print("-- Totally draw %d labeled samples" % len(dset.targets))
        print("-- The labeled dist is", counts / np.sum(counts))
    
    return dset
        

            

def GetData(dbname, data_root, train=True):
    "Get dataset from data root."
    
    if dbname == 'MNIST':
        trans = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()
        ])
        dataset = dsets.MNIST(data_root, train=train, transform=trans, download=True)

    elif dbname == 'SVHN':
        trans = transforms.Compose([
            transforms.Resize(37),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])
        split = 'train' if train else 'test'
        dataset = dsets.SVHN(dataset, split=split, transform=trans, download=True)
    
    else:
        raise NotImplementedError

    return dataset