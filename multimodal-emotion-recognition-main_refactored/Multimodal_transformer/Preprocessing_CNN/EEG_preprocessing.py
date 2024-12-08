import torch
import torch.nn as nn
from torch import fft


class EEGCNNPreprocessor(nn.Module):
    def __init__(self, num_channels=14, d_model=128, cnn_out_channels=32):
        super(EEGCNNPreprocessor, self).__init__()
        self.d_model = d_model

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(64, cnn_out_channels, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, d_model))  # Final spatial pooling
        )

        self.fft_projector = nn.Linear(d_model,d_model)


    def forward(self, x):
        """
        Args:
            x: EEG input tensor of shape (batch_size, num_channels, seq_length).
               num_channels: number of EEG channels (e.g., 14)
               seq_length: number of timesteps per channel.
        Returns:
            Processed EEG features: shape (batch_size, seq_length, d_model).
        """
        
        # Apply CNN for spatial feature extraction
        cnn_input = x.unsqueeze(1)  # Add a channel dimension for CNN (B, 1, C, T)
        cnn_features = self.cnn(cnn_input).squeeze(2)  # (B, cnn_out_channels, seq_length)

        # Apply FFT along the time axis
        fft_features = fft.rfft(x, dim=-1)  # (B, num_channels, freq_bins)
        fft_magnitude = torch.abs(fft_features)  # Magnitude spectrum (B, num_channels, freq_bins)

        # Average across the frequency bins to reduce dimensionality and align with CNN output
        fft_magnitude = fft_magnitude.mean(dim=-1)  # (B, num_channels)

        # Project FFT features to match cnn_out_channels
        fft_projected = self.fft_projector(fft_magnitude)  # (B, seq_length, d_model)

        # Combine CNN features and FFT features
        final_features = cnn_features + fft_projected  # (B, seq_length, d_model)
       
        return final_features