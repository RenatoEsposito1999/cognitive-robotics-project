import torch
import torch.nn as nn
from torch import fft


class EEGCNNPreprocessor(nn.Module):
    def __init__(self, num_channels=14, d_model=128, cnn_out_channels=32, seq_reduction_factor=2):
        super(EEGCNNPreprocessor, self).__init__()
        self.d_model = d_model
        self.seq_reduction_factor = seq_reduction_factor

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(num_channels, 1), stride=1, padding=(0, 0)),  # Adjust kernel size
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(64, cnn_out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.ReLU()
        )

        # Sequence reduction via 1D convolution
        self.seq_reduction = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=self.seq_reduction_factor, stride=self.seq_reduction_factor)

        # Projection layer
        self.cnn_projector = nn.Linear(cnn_out_channels, d_model)

        # FFT projection
        self.fft_projector = nn.Linear(num_channels, d_model)  # Adjust input size to num_channels

    def forward(self, x):
        """
        Args:
            x: EEG input tensor of shape (batch_size, seq_length, num_channels).
               num_channels: number of EEG channels (e.g., 14)
               seq_length: number of timesteps per channel.
        Returns:
            Processed EEG features: shape (batch_size, reduced_seq_length, d_model).
        """
        batch_size, seq_length, num_channels = x.size()

        # Apply CNN for spatial feature extraction
        cnn_input = x.permute(0, 2, 1).unsqueeze(1)  # Reshape to (B, 1, num_channels, seq_length)
        cnn_features = self.cnn(cnn_input).squeeze(2)  # (B, cnn_out_channels, seq_length)
        cnn_features = cnn_features.permute(0, 2, 1)  # (B, seq_length, cnn_out_channels)

        # Reduce sequence length with 1D convolution
        cnn_features = cnn_features.permute(0, 2, 1)  # (B, cnn_out_channels, seq_length)
        cnn_features = self.seq_reduction(cnn_features)  # (B, cnn_out_channels, reduced_seq_length)
        cnn_features = cnn_features.permute(0, 2, 1)  # (B, reduced_seq_length, cnn_out_channels)

        # Project CNN features to d_model
        cnn_features = self.cnn_projector(cnn_features)  # (B, reduced_seq_length, d_model)

        # Apply FFT along the time axis
        fft_features = fft.rfft(x, dim=1)  # (B, freq_bins, num_channels)
        fft_magnitude = torch.abs(fft_features)  # Magnitude spectrum (B, freq_bins, num_channels)

        # Average across the frequency bins to reduce dimensionality
        fft_magnitude = fft_magnitude.mean(dim=1)  # (B, num_channels)

        # Project FFT features to match d_model
        fft_projected = self.fft_projector(fft_magnitude)  # (B, d_model)

        # Expand FFT features to align with reduced_seq_length
        reduced_seq_length = cnn_features.size(1)
        fft_projected = fft_projected.unsqueeze(1).expand(-1, reduced_seq_length, -1)  # (B, reduced_seq_length, d_model)

        # Combine CNN features and FFT features
        final_features = cnn_features + fft_projected  # (B, reduced_seq_length, d_model)

        #print(f"Final features shape: {final_features.shape}")
        return final_features
