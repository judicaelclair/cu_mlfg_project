import torch
import torch.nn as nn
import timeit
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Squeeze(nn.Module):
  def forward(self, x):
    return x.squeeze()

class Unsqueeze(nn.Module):
  def __init__(self, *dims):
    super(Unsqueeze, self).__init__()
    self.dims = dims
      
  def forward(self, x):
    return x.unsqueeze(*self.dims)

  def __repr__(self) -> str:
      return "Unsqueeze({})".format(",".join([str(d) for d in self.dims]))

def compute_kl_loss(mu, logvar):
  # see: https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
  return torch.sum(torch.pow(mu, 2) + torch.exp(logvar) - 1 - logvar) / 2

def compute_accuracy(y_pred, y, threshold = 1):
  return torch.mean((torch.abs(y_pred - y) < threshold).float()).detach().cpu().numpy()

def train_model_(run_single_epoch, model, train_dataset, validation_dataset, test_dataset = None, epochs = 100000, patience = 1000, verbose = True, batch_size = 32, alpha = None):
  model.to(torch_device)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
  validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
  optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

  train_kl_losses = []
  val_kl_losses = []
  train_rec_losses = []
  val_rec_losses = []
  train_losses = []
  val_losses = []
  train_accs = []
  val_accs = []
  rem_patience = patience
  best_val_loss = np.inf
  checkpoint_fn = 'model_checkpoint.pt'
  
  for epoch in range(epochs):
    start_time = timeit.default_timer()
    
    train_kl_loss, train_rec_loss, train_loss, train_acc = run_single_epoch(train_dataset, train_dataloader, model, optimizer, alpha)
    val_kl_loss, val_rec_loss, val_loss, val_acc = run_single_epoch(validation_dataset, validation_dataloader, model, None, alpha)
    
    train_kl_losses.append(train_kl_loss)
    val_kl_losses.append(val_kl_loss)
    train_rec_losses.append(train_rec_loss)
    val_rec_losses.append(val_rec_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    if val_loss < best_val_loss: 
      torch.save(model.state_dict(), checkpoint_fn)
      best_val_loss = val_loss
      rem_patience = patience
    else: 
      rem_patience -= 1
      if rem_patience <= 0:
        break
    
    elapsed = float(timeit.default_timer() - start_time)
    if verbose:
      print("Epoch {} took {:.2f}s. Train kl_loss: {:.4f} rec_loss: {:.4f} loss: {:.4f} acc: {:.4f}. Val kl_loss: {:.4f} rec_loss: {:.4f} loss: {:.4f} acc: {:.4f}. Patience left: {}".format(
            epoch+1, elapsed, train_kl_loss, train_rec_loss, train_loss, train_acc, val_kl_loss, val_rec_loss, val_loss, val_acc, rem_patience))
    
    if test_dataset is not None and val_loss == best_val_loss:
      test_kl_loss, test_rec_loss, test_loss, test_acc = run_single_epoch(train_dataset, train_dataloader, model)
      print("Train kl_loss: {:.4f} rec_loss: {:.4f} loss: {:.4f} acc: {:.4f}".format(test_kl_loss, test_rec_loss, test_loss, test_acc))
      test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
      test_kl_loss, test_rec_loss, test_loss, test_acc = run_single_epoch(test_dataset, test_dataloader, model)
      print("Test kl_loss: {:.4f} rec_loss: {:.4f} loss: {:.4f} acc: {:.4f}".format(test_kl_loss, test_rec_loss, test_loss, test_acc))
  
  # recover best model
  model.load_state_dict(torch.load(checkpoint_fn))
          
  if verbose:
    best_idx = np.argmax(val_losses)
    print("Best model index is {}. Train kl_loss: {:.4f} rec_loss: {:.4f} loss: {:.4f} acc: {:.4f}. Val kl_loss: {:.4f} rec_loss: {:.4f} loss: {:.4f} acc: {:.4f}.".format(
          best_idx, train_kl_losses[best_idx], train_rec_losses[best_idx], train_losses[best_idx], train_accs[best_idx], val_kl_losses[best_idx], val_rec_losses[best_idx], val_losses[best_idx], val_accs[best_idx]))
    
    if test_dataset is not None:
      test_kl_loss, test_rec_loss, test_loss, test_acc = run_single_epoch(train_dataset, train_dataloader, model)
      print("Train kl_loss: {:.4f} rec_loss: {:.4f} loss: {:.4f} acc: {:.4f}".format(test_kl_loss, test_rec_loss, test_loss, test_acc))
      test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
      test_kl_loss, test_rec_loss, test_loss, test_acc = run_single_epoch(test_dataset, test_dataloader, model)
      print("Test kl_loss: {:.4f} rec_loss: {:.4f} loss: {:.4f} acc: {:.4f}".format(test_kl_loss, test_rec_loss, test_loss, test_acc))
        
  return train_kl_losses, val_kl_losses, train_rec_losses, val_rec_losses, train_losses, val_losses, train_accs, val_accs
