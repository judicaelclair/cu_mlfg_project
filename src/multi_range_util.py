import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

from common_util import DATA_DIR, Squeeze, Unsqueeze, torch_device, compute_kl_loss, compute_accuracy, train_model_

class ChromGroupedDataset(torch.utils.data.Dataset):
  def __init__(self, atac_path = None, dname_path = None):
    if atac_path is None:
      atac_path = os.path.join(DATA_DIR, 'atac_chrom_grouped.json')
    if dname_path is None:
      dname_path = os.path.join(DATA_DIR, 'dname_chrom_grouped.json')
    
    with open(atac_path, 'r') as file:
      self.atac_json = json.loads(file.read())
    
    with open(dname_path, 'r') as file:
      self.dname_json = json.loads(file.read())
    
    self.in_len = 42
    self.out_len = 42
    self.max_num_ranges = 8
    self.dist_threshold = 3000
    
    self.y = []
    self.sample_num_ranges = []
    self.subrange_dname_seqs = []
    self.subrange_dname_lengths = []
    for chrom_idx, chr_atac_vs in self.atac_json.items():
      low_idx = 0
      high_idx = 1
      keys = list(chr_atac_vs.keys())
      while low_idx < len(keys):
        while high_idx < len(keys) and (chr_atac_vs[keys[high_idx-1]]["data"]["dist0_1"] < self.dist_threshold) and ((high_idx - low_idx) < self.max_num_ranges):
          high_idx += 1
        
        self.subrange_dname_seqs.append([])
        self.subrange_dname_lengths.append([])
        self.y.append([])
        
        atac_chrom_json = self.atac_json[chrom_idx]
        dname_chrom_json = self.dname_json[chrom_idx]
        dname_chrom_keys = list(dname_chrom_json.keys())
        for j in range(low_idx, high_idx):
          range_key = dname_chrom_keys[j]
          kvs = dname_chrom_json[range_key]["data"]
          points = [v for k, v in kvs.items() if k.startswith("point")]
          self.subrange_dname_seqs[-1].append(torch.from_numpy(np.array(points, dtype=np.float32)))
          self.subrange_dname_lengths[-1].append(len(points))
          assert len(points) > 0
          
          self.y[-1].append(atac_chrom_json[range_key]["data"]["point0"])
        
        self.sample_num_ranges.append(high_idx - low_idx)
        low_idx = high_idx
        high_idx += 1
    
    print("distribution (# ranges in sample, # occurences):")
    print(np.array(np.unique(self.sample_num_ranges, return_counts=True)).T)
    print()

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return idx
  
  def generate_batch(self, batch_idxs):    
    y = []
    subrange_dname_seqs = []
    subrange_dname_lengths = []
    sample_num_ranges = []
    
    for i in batch_idxs:
      y += self.y[i]
      subrange_dname_seqs += self.subrange_dname_seqs[i]
      subrange_dname_lengths += self.subrange_dname_lengths[i]
      sample_num_ranges.append(self.sample_num_ranges[i])
    
    # see:
    # [1] https://stackoverflow.com/a/69968933
    # [2] https://stackoverflow.com/a/51030945
    ps = torch.nn.utils.rnn.pad_sequence(subrange_dname_seqs, batch_first=True)
    pps = torch.nn.utils.rnn.pack_padded_sequence(ps, lengths=subrange_dname_lengths, batch_first=True, enforce_sorted=False)
    x = pps.to(torch_device)
    
    y = torch.from_numpy(np.array(y, dtype=np.float32)).to(torch_device)
    
    return x, y, subrange_dname_lengths, sample_num_ranges

def load_split_dataset(train = 0.7, validation = 0.15):
  dataset = ChromGroupedDataset()

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
    x, y, subrange_dname_lengths, sample_num_ranges = subset.dataset.generate_batch(batch_idxs)

    out = model(x, sample_num_ranges)
    
    is_vae = isinstance(out, tuple)
    if is_vae:
      y_pred, mu, logvar = out
      kl_loss = compute_kl_loss(mu, logvar) 
    else:
      y_pred = out
      kl_loss = torch.tensor(0)

    y_pred = y_pred.squeeze()
    
    if preserve_weighting and (not training):
      y_pred = torch.from_numpy(np.repeat(y_pred.detach().cpu().numpy(), subrange_dname_lengths, axis=0))
      y = torch.from_numpy(np.repeat(y.detach().cpu().numpy(), subrange_dname_lengths, axis=0))
      batch_sizes.append(np.sum(subrange_dname_lengths))
    
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
