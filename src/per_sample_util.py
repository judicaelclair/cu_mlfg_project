import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

from common_util import DATA_DIR, Squeeze, Unsqueeze, torch_device, compute_kl_loss, compute_accuracy, train_model_


def load_dataset(center_y = False):
  Y = torch.tensor(pd.read_csv(os.path.join(DATA_DIR, 'atacseq_processed.csv'), sep=',').drop(columns='Composile Element REF').values.astype(np.float32), device=torch_device)[:, 1:]
  X = torch.tensor(pd.read_csv(os.path.join(DATA_DIR, 'methylation_processed.csv'), sep=',').drop(columns='Composile Element REF').drop(columns='chromEnd').values.astype(np.float32), device=torch_device)[:, 1:]

  if center_y:
    y_min = torch.min(Y.flatten())
    y_max = torch.max(Y.flatten())
    Y = (Y-y_min)/(y_max-y_min)
  
  dataset = torch.utils.data.TensorDataset(X, Y)
  
  if center_y:
    return dataset, X.shape[-1], Y.shape[-1], y_min.detach().cpu().numpy(), y_max.detach().cpu().numpy()
  else:
    return dataset, X.shape[-1], Y.shape[-1]

def load_split_dataset(train = 0.7, validation = 0.15, center_y = False):
  if center_y:
    dataset, in_len, out_len, y_min, y_max = load_dataset(center_y=True)
  else:
    dataset, in_len, out_len = load_dataset(center_y=False)

  train_size = int(train * len(dataset))
  validation_size = int(validation * len(dataset))
  test_size = len(dataset) - train_size - validation_size
  train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

  if center_y:
    return train_dataset, validation_dataset, test_dataset, in_len, out_len, y_min, y_max
  else:
    return train_dataset, validation_dataset, test_dataset, in_len, out_len

def run_single_epoch(subset, dataloader, model, optimizer = None, alpha = None):
  alpha = 0.005 if alpha is None else alpha
  training = optimizer is not None
  torch.set_grad_enabled(training)
  model.train() if training else model.eval() 
  
  kl_losses = []
  rec_losses = []
  losses = []
  accuracies = []
  
  for x, y in dataloader:
    x = x.to(torch_device)
    y = y.to(torch_device)

    out = model(x)
    
    is_vae = isinstance(out, tuple)
    if is_vae:
      y_pred, mu, logvar = out
      kl_loss = compute_kl_loss(mu, logvar) 
    else:
      y_pred = out
      kl_loss = torch.tensor(0)

    y_pred = y_pred.squeeze()
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
  
  return np.mean(kl_losses), np.mean(rec_losses), np.mean(losses), np.mean(accuracies)

def train_model(*args, **kwargs):
  return train_model_(run_single_epoch, *args, **kwargs)
