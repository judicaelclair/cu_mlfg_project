import torch
import torch.nn as nn

import per_sample_util as util
    
class VAE(nn.Module):
    def __init__(self,
                 output_length,
                 first_layer_num_channels = 32,
                 latent_dim = 4,
                 dropout = 0.2):
        super(VAE, self).__init__()

        self.output_length = output_length
        self.first_layer_num_channels = first_layer_num_channels
        self.latent_dim = latent_dim
        
        activation = nn.ELU(inplace=True)
        
        num_layers = 5
        size_before_latent = self.first_layer_num_channels * (2 ** 2)

        # Encoder
        encoder_layers = [util.Squeeze(), util.Unsqueeze(1)]
        for i in range(num_layers):
            num_in_chans = 1 if i == 0 else (self.first_layer_num_channels * (2 ** (i - 1)))
            num_out_chans = self.first_layer_num_channels * (2 ** i)
            encoder_layers += [nn.Conv1d(num_in_chans, num_out_chans, kernel_size=4, padding=1, stride=2),
                               nn.BatchNorm1d(num_out_chans),
                               activation]
        
        encoder_layers += [util.Squeeze(),
                           nn.Linear(self.first_layer_num_channels * (2 ** (num_layers - 1)),
                                     size_before_latent * 2),
                           nn.BatchNorm1d(size_before_latent * 2),
                           nn.Dropout(dropout),
                           activation,
                           nn.Linear(size_before_latent * 2,
                                     size_before_latent),
                           nn.BatchNorm1d(size_before_latent),
                           nn.Dropout(dropout),
                           activation]
        
        self.encoder_net = nn.Sequential(*encoder_layers)

        self.encoder_left_branch = nn.Sequential(nn.Linear(size_before_latent, self.latent_dim), 
                                                 nn.Dropout(dropout))
        
        self.encoder_right_branch = nn.Sequential(nn.Linear(size_before_latent, self.latent_dim),
                                                  nn.Dropout(dropout))

        # Decoder
        decoder_layers = []
        
        decoder_layers += [nn.Linear(self.latent_dim,
                                     size_before_latent),
                           nn.BatchNorm1d(size_before_latent),
                           nn.Dropout(dropout),
                           activation,
                           nn.Linear(size_before_latent,
                                     size_before_latent * 2),
                           nn.BatchNorm1d(size_before_latent * 2),
                           nn.Dropout(dropout),
                           activation,
                           nn.Linear(size_before_latent * 2,
                                     self.first_layer_num_channels * (2 ** (num_layers - 1))),
                           nn.BatchNorm1d(self.first_layer_num_channels * (2 ** (num_layers - 1))),
                           nn.Dropout(dropout),
                           activation,
                           util.Unsqueeze(-1)]
        
        for i in reversed(range(num_layers)):
            num_in_chans = self.first_layer_num_channels * (2 ** i)
            num_out_chans = 1 if i == 0 else (self.first_layer_num_channels * (2 ** (i - 1)))
            decoder_layers += [nn.ConvTranspose1d(num_in_chans, num_out_chans, kernel_size=4, padding=1, stride=2),
                               nn.BatchNorm1d(num_out_chans),
                               activation]
        
        decoder_layers += [util.Squeeze(),
                           nn.Linear(32, self.output_length)]
        
        self.decoder_net = nn.Sequential(*decoder_layers)

    def sample(self, mu, logvar):
        if self.training:
            # see: https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
            return torch.distributions.Normal(mu, torch.exp(0.5 * logvar)).rsample()
        else:
            return mu

    def encode(self, x):
        out = self.encoder_net(x)
        return self.encoder_left_branch(out), self.encoder_right_branch(out)

    def decode(self, z):
        out = self.decoder_net(z)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        y = self.decode(z)
        return y, mu, logvar

if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset, in_len, out_len = util.load_split_dataset()
    model = VAE(out_len)
    print(model)
    util.train_model(model, train_dataset, validation_dataset, test_dataset)
