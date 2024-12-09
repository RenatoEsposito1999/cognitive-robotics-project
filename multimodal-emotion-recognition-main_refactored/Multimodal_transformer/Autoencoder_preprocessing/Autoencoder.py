import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class EEGAutoencoder(nn.Module):
    def __init__(self,input_dim=14,chunk_size=1000, latent_dim=128):
        super(EEGAutoencoder, self).__init__()
        self.chunk_size = chunk_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder and Decoder for smaller chunks
        self.chunk_encoder = nn.Sequential(
            nn.Linear(chunk_size * input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.chunk_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, chunk_size * input_dim),
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        assert input_dim == self.input_dim

        # Split sequence into chunks
        x_chunks = x.view(batch_size, -1, self.chunk_size, input_dim)  # Shape: [batch_size, num_chunks, chunk_size, input_dim]
        x_chunks = x_chunks.flatten(0, 1)  # Combine batch and num_chunks for processing

        # Process each chunk
        x_chunks = x_chunks.view(x_chunks.size(0), -1)  # Flatten each chunk
        latent = self.chunk_encoder(x_chunks)  # Encode chunks
        reconstructed_chunks = self.chunk_decoder(latent)  # Decode chunks

        # Reshape reconstructed chunks
        reconstructed_chunks = reconstructed_chunks.view(batch_size, -1, self.chunk_size, input_dim)
        reconstructed = reconstructed_chunks.flatten(1, 2)  # Combine chunks back into full sequence

        return latent, reconstructed

