import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import shutil
from torch.utils.data import DataLoader

 
class EEGAutoencoder(nn.Module):
    def __init__(self, num_channels=14, num_bands=5, latent_channels=64, batch_size=1):
        super(EEGAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=(3, num_bands), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_channels, kernel_size=(3, num_bands), padding=(1, 0))
        )

    def forward(self, x):
        latent = self.encoder(x).squeeze(-1)  # (batch_size, latent_channels, seq_length)
        reconstructed = self.decoder(latent.unsqueeze(-1))
        return latent, reconstructed
        