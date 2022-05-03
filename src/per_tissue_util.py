import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

from common_util import DATA_DIR, Squeeze, Unsqueeze, torch_device, compute_kl_loss, compute_accuracy, train_model_

class TissueGroupedDataset(torch.utils.data.Dataset):
  def __init__(self, atac_path = None, dname_path = None):
    if atac_path is None:
      atac_path = os.path.join(DATA_DIR, 'atac_tissue_grouped.json')
    if dname_path is None:
      dname_path = os.path.join(DATA_DIR, 'dname_tissue_grouped.json')
    
    with open(atac_path, 'r') as file:
      self.atac_json = json.loads(file.read())
    
    with open(dname_path, 'r') as file:
      self.dname_json = json.loads(file.read())
    
    self.length = len(self.atac_json)
    self.in_len = 1
    self.out_len = 1
    self.keys = list(self.atac_json.keys())
    
    lengths = {}
    for key in self.atac_json:
      length = len(self.atac_json[key])
      if length in lengths:
        lengths[length] += 1
      else:
        lengths[length] = 1

    # lengths
    # print(np.array(list(lengths.items()))[np.array(list(lengths.keys())).argsort()])
    
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return idx

  def generate_batch(self, batch_idxs):
    dname_seqs = []
    atac_seqs = []
    lengths = []
    for i in batch_idxs:
      key = self.keys[i]
      dname_seqs.append(torch.from_numpy(np.array(self.dname_json[key], dtype=np.float32)).unsqueeze(1))
      atac_seqs.append(torch.from_numpy(np.array(self.atac_json[key], dtype=np.float32)).reshape(-1, 1))
      lengths.append(len(self.atac_json[key]))
      assert lengths[-1] > 0
    
    enforce_sorted = False
    if enforce_sorted:
      sort_idxs = np.argsort(lengths)[::-1]
      lengths = [lengths[i] for i in sort_idxs]
      dname_seqs = [dname_seqs[i] for i in sort_idxs]
      atac_seqs = [atac_seqs[i] for i in sort_idxs]

    # see:
    # [1] https://stackoverflow.com/a/69968933
    # [2] https://stackoverflow.com/a/51030945
    dname_ps = torch.nn.utils.rnn.pad_sequence(dname_seqs, batch_first=True)
    x = torch.nn.utils.rnn.pack_padded_sequence(dname_ps, lengths=lengths, batch_first=True, enforce_sorted=enforce_sorted).to(torch_device)
    y = torch.cat(atac_seqs, 0).to(torch_device)
    
    return x, y, lengths

def load_split_dataset(train = 0.7, validation = 0.15):
  dataset = TissueGroupedDataset()

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
  
  for batch_idxs in dataloader:
    x, y, lengths = subset.dataset.generate_batch(batch_idxs)
    batch_sizes.append(np.sum(lengths))

    out = model(x)
    
    is_vae = isinstance(out, tuple)
    if is_vae:
      y_pred, mu, logvar = out
      kl_loss = compute_kl_loss(mu, logvar) 
    else:
      y_pred = out
      kl_loss = torch.tensor(0)
      
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
