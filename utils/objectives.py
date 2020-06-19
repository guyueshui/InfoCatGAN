import numpy as np 
import torch
import torch.nn.functional as F 

def CategoricalCrossentropy(predictions, targets_dist, epsilon=1e-6):
    predictions = predictions.squeeze()
    targets_dist = targets_dist.squeeze()
    assert predictions.size() == targets_dist.size(), "input shape mismatch"
    ce = -predictions*torch.log(targets_dist + epsilon)
    if predictions.dim() == 2:
        ce = ce.sum(dim=1).mean()
    elif predictions.dim() == 1:
        ce = ce.sum()
    else:
        raise NotImplementedError
    return ce


def Entropy(distribution, epsilon=1e-6):
    return CategoricalCrossentropy(distribution, distribution, epsilon)


def CategoricalCrossentropyOfMean(predictions):
    nclass = predictions.size(1)
    uniform_targets = torch.ones(1, nclass, device=predictions.device) / nclass
    pred = predictions.mean(dim=0)
    pred = pred / pred.sum()
    return CategoricalCrossentropy(pred, uniform_targets)


def CategoricalCrossentropySslSeparated(predictions_l, targets,
        predictions_u, alpha_labeled=1., alpha_unlabeled=.3, alpha_average=1e-3):
    # for labeled data, minimize CE[p_c(y), y_l]
    celoss = F.cross_entropy(predictions_l, targets).mean()
    # for unlabeled data, minimize H(p(y|x_u))
    enloss = Entropy(predictions_u)
    # for balanced on unlabeled data
    avloss = CategoricalCrossentropyOfMean(predictions_u)
    # THE above forms confidence loss in paper, from CatGAN.
    return alpha_labeled*celoss + alpha_unlabeled*enloss + alpha_average*avloss

def CategoricalCrossentropyUslSeparated(predictions, alpha=.3, alpha_average=1e-3):
    enloss = Entropy(predictions)
    avloss = CategoricalCrossentropyOfMean(predictions)
    return alpha*enloss + alpha_average*avloss

def CategoricalAccuracy(predictions, targets, map_to_real=None):
    predictions = predictions.squeeze().cpu().detach().numpy()
    targets = targets.squeeze().detach().cpu().detach().numpy()
    if predictions.ndim == 2:
        pred = np.argmax(predictions, axis=-1)
    elif predictions.ndim == 1:
        pred = predictions
    else:
        raise ValueError("undesired input shape")
    if map_to_real is not None:
        pred = map_to_real[pred]
    acc = np.equal(pred, targets).mean()
    return acc

def CategoryMatching(predictions, targets, num_class):
    predictions = predictions.detach().cpu().squeeze().numpy()
    targets = targets.detach().cpu().squeeze().numpy()
    assert predictions.shape == targets.shape
    mat = np.zeros((num_class, num_class), dtype=int)
    for i in range(len(targets)):
        mat[predictions[i], targets[i]] += 1
    print(mat)
    map_to_real = np.argmax(mat, axis=1)
    print('map_to_real is', map_to_real)
    return map_to_real

def LogGaussian(x, mu, var, epsilon=1e-6):
    logli = -0.5 * (var.mul(2*np.pi) + epsilon).log() - \
            (x-mu).pow(2).div(var.mul(2.0) + epsilon)
    if (logli >= 0).any():
        logli = torch.clamp(logli, max=0-epsilon)
    assert (logli < 0).all(), "log of probability must < 0"
    return logli.sum(1).mean().mul(-1)

