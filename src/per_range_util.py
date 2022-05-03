import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

from common_util import DATA_DIR, Squeeze, Unsqueeze, torch_device, compute_kl_loss, compute_accuracy, train_model_

class RangeGroupedDataset(torch.utils.data.Dataset):
  def __init__(self, atac_path = None, dname_path = None):
    if atac_path is None:
      atac_path = os.path.join(DATA_DIR, 'atac_range_grouped.json')
    if dname_path is None:
      dname_path = os.path.join(DATA_DIR, 'dname_range_grouped.json')
    
    with open(atac_path, 'r') as file:
      self.atac_json = json.loads(file.read())
    
    with open(dname_path, 'r') as file:
      self.dname_json = json.loads(file.read())
    
    self.length = len(self.atac_json)
    self.in_len = len(self.dname_json["0"]["point0"]) # + 1
    self.out_len = len(self.atac_json["0"]["point0"])
    
    lengths = {}
    for key in self.dname_json:
      kvs = self.dname_json[key]
      points = [v for k, v in kvs.items() if k.startswith("point")]
      length = len(points)
      if length in lengths:
        lengths[length] += 1
      else:
        lengths[length] = 1
    
    # lengths
    # print(np.array(list(lengths.items()))[np.array(list(lengths.keys())).argsort()])
  
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    atac = torch.tensor(self.atac_json[str(idx)]["point0"], device = torch_device)
    return idx, atac

  def generate_batch(self, batch_idxs):
    s = []
    lengths = []
    for i in batch_idxs:
      kvs = self.dname_json[str(int(i))]
      points = [v for k, v in kvs.items() if k.startswith("point")]
      # distances = [0] + [v for k, v in kvs.items() if k.startswith("dist")]
      # s.append(torch.from_numpy(np.array([[distances[j]] + p for j, p in enumerate(points)], dtype=np.float32)))
      s.append(torch.from_numpy(np.array([p for j, p in enumerate(points)], dtype=np.float32)))
      lengths.append(len(points))
      assert len(points) > 0
    
    enforce_sorted = False
    if enforce_sorted:
      sort_idxs = np.argsort(lengths)[::-1]
      lengths = [lengths[i] for i in sort_idxs]
      s = [s[i] for i in sort_idxs]

    # see:
    # [1] https://stackoverflow.com/a/69968933
    # [2] https://stackoverflow.com/a/51030945
    ps = torch.nn.utils.rnn.pad_sequence(s, batch_first=True)
    pps = torch.nn.utils.rnn.pack_padded_sequence(ps, lengths=lengths, batch_first=True, enforce_sorted=enforce_sorted).to(torch_device)
    return pps, lengths

def load_split_dataset(train = 0.7, validation = 0.15):
  dataset = RangeGroupedDataset()

  train_size = int(train * len(dataset))
  validation_size = int(validation * len(dataset))
  test_size = len(dataset) - train_size - validation_size
  train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
  
  return train_dataset, validation_dataset, test_dataset

def run_single_epoch(subset, dataloader, model, optimizer = None, alpha = None, preserve_weighting = True):
  alpha = 0.005 if alpha is None else alpha
  training = optimizer is not None
  torch.set_grad_enabled(training)
  model.train() if training else model.eval() 
  
  kl_losses = []
  rec_losses = []
  losses = []
  accuracies = []
  batch_sizes = []
  
  for x, y in dataloader:
    x, x_lengths = subset.dataset.generate_batch(x)

    out = model(x)
    
    is_vae = isinstance(out, tuple)
    if is_vae:
      y_pred, mu, logvar = out
      kl_loss = compute_kl_loss(mu, logvar) 
    else:
      y_pred = out
      kl_loss = torch.tensor(0)

    y_pred = y_pred.squeeze()
    
    if preserve_weighting and (not training):
      y_pred = torch.from_numpy(np.repeat(y_pred.detach().cpu().numpy(), x_lengths, axis=0))
      y = torch.from_numpy(np.repeat(y.detach().cpu().numpy(), x_lengths, axis=0))
      batch_sizes.append(np.sum(x_lengths))
      
    rec_loss = F.mse_loss(y_pred, y)
    loss = rec_loss + alpha * kl_loss

    if training:
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    kl_losses.append(kl_loss.detach().cpu().numpy())
    rec_losses.append(rec_loss.detach().cpu().numpy())
    losses.append(loss.detach().cpu().numpy())
    accuracies.append(compute_accuracy(y_pred, y))
  
  if preserve_weighting and (not training):
    return np.average(kl_losses, weights=batch_sizes), np.average(rec_losses, weights=batch_sizes), np.average(losses, weights=batch_sizes), np.average(accuracies, weights=batch_sizes)
  else:
    return np.mean(kl_losses), np.mean(rec_losses), np.mean(losses), np.mean(accuracies)

def train_model(*args, **kwargs):
  return train_model_(run_single_epoch, *args, **kwargs)
