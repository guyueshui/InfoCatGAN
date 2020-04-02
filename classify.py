import torch
import numpy as np

from utils import get_data
from utils import CustomDataset
from config import get_config
from CatGAN import CatGAN

class Classifier:
  def __init__(self, model, model_state, dataset):
    """
    Args:
    -- model: the model to perform classify task
    -- model_state (str): full path of model checkpoint
    -- dataset: dataset used to perform classification 
    """
    self.model = model
    self.dataset = dataset
    model.load_model(model_state, *model.models)

  def classify(self):
    def _batch_classify(imgs, map_to_real=None):
      if map_to_real is not None:
        assert len(map_to_real) == self.model.cat_dim
        assert np.max(map_to_real) < self.model.cat_dim
        assert np.min(map_to_real) >= 0
      else:
        map_to_real = np.arange(self.model.cat_dim)

      logits = self.model.raw_classify(imgs)
      fake_labels = np.argmax(logits, axis=1)
      predicted = map_to_real[fake_labels].reshape(-1)
      return predicted
      
    # prepare data
    loader = torch.utils.data.DataLoader(self.dataset, 
             batch_size=100, shuffle=False, num_workers=4)
    map_to_real = self.get_map_to_real()

    num_correct = 0
    for num_iter, (images, labels) in enumerate(loader):
      predicted = _batch_classify(images, map_to_real)
      with torch.no_grad():
        labels = labels.numpy()

      assert predicted.shape == labels.shape
      loss = abs(predicted - labels)
      _, cnt = np.unique(loss, return_counts=True)
      batch_num_correct = cnt[0]
      # print('Matched: ', batch_num_correct)
      num_correct += batch_num_correct
      
    acc = num_correct / len(dataset)
    print('Accuracy is: ', acc)
  
  def get_map_to_real(self):
    if self.model.config.gan_type == 'ssinfogan':
      return None
    dset = CustomDataset(self.dataset, 0.01)
    dset.report()
    labeled_set = torch.utils.data.DataLoader(dset.labeled, batch_size=100)
    abatch = next(iter(labeled_set))
    images, labels = abatch
    logits = self.model.raw_classify(images)
    yk = np.argmax(logits, axis=1).reshape(-1)
    labels = labels.numpy().reshape(-1)
    assert len(yk) == len(labels)
    mat = np.zeros((self.model.cat_dim, self.model.cat_dim), dtype=int)
    for i in range(len(yk)):
      mat[yk[i], labels[i]] += 1
    print(mat)
    map_to_real = np.argmax(mat, axis=1)
    print("map_to_real is ", map_to_real)
    return map_to_real


args = get_config()
path = 'results/' + args.dataset
path += '/catgan'
path += '/model-epoch-100.pt'

dataset = get_data(args.dataset, args.data_root, train=False)
gan = CatGAN(args, dataset)
gan.load_model(path, gan.D)

if __name__ == '__main__':
  # Use test dataset.
  loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1)

  # Category matching step of CatGAN.
  dset = CustomDataset(dataset, 0.01)
  dset.report()
  labeled_set = torch.utils.data.DataLoader(dset.labeled, batch_size=100)
  abatch = next(iter(labeled_set))
  images, labels = abatch
  images = images.to(gan.device)
  with torch.no_grad():
    _, logits = gan.D(images)
    logits = logits.cpu().numpy()
    yk = np.argmax(logits, axis=1).reshape(-1)
    labels = labels.numpy().reshape(-1)
  assert len(yk) == len(labels)
  mat = np.zeros((10, 10), dtype=int)
  for i in range(len(yk)):
    mat[yk[i], labels[i]] += 1
  print(mat)
  map_to_real = np.argmax(mat, axis=1)
  map_to_real[4] = 7
  map_to_real[2] = 9 
  print("map_to_real is ", map_to_real)

  num_correct = 0
  for num_iter, (images, labels) in enumerate(loader):
    images = images.to(gan.device)
    predicted = gan.classify(images, map_to_real)
    with torch.no_grad():
      labels = labels.numpy()

    assert predicted.shape == labels.shape
    loss = abs(predicted - labels)
    _, cnt = np.unique(loss, return_counts=True)
    batch_num_correct = cnt[0]
    print('Matched: ', batch_num_correct)
    num_correct += batch_num_correct
    
  acc = num_correct / len(dataset)
  print('Accuracy is: ', acc)
