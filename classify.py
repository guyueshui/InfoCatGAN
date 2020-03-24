import torch
import numpy as np

from utils import get_data
from config import get_config
from SS_InfoGAN import SS_InfoGAN

args = get_config()
path = 'results/' + args.dataset
path += '/re-ssinfogan'
path += '/model-epoch-70.pt'

dataset = get_data(args.dataset, args.data_root, train=False)
gan = SS_InfoGAN(args, dataset)
gan.load_model(path, gan.FD, gan.Q)

def Classify(imgs):
  with torch.no_grad():
    logits, _, _ = gan.Q( gan.FD(imgs) )
    logits = logits.cpu().numpy()
  predicted = np.argmax(logits, axis=1)
  return predicted

if __name__ == '__main__':
  # Use test dataset.
  print(len(dataset))
  loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=1)
  abatch = next(iter(loader))
  images, labels = abatch

  num_correct = 0
  for num_iter, (images, labels) in enumerate(loader):
    images = images.to(gan.device)
    predicted = Classify(images)
    with torch.no_grad():
      labels = labels.numpy()

    loss = abs(predicted - labels)
    _, cnt = np.unique(loss, return_counts=True)
    batch_num_correct = cnt[0]
    print('Matched: ', batch_num_correct)
    num_correct += batch_num_correct
    
  acc = num_correct / len(dataset)
  print('Accuracy is: ', acc)
