import torch
import torch.nn as nn

class EEGCNNPreprocessor(nn.Module):
    def __init__(self, num_channels=14, d_model=128, cnn_out_channels=128):
        super(EEGCNNPreprocessor, self).__init__()
        self.d_model = d_model

        # Spatial CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(num_channels, 1), stride=1, padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.ReLU()
        )

        # Temporal Convolution
        self.temporal_conv = nn.Conv2d(d_model, d_model, kernel_size=(1, 5), stride=1, padding=(0, 2))
        
        # Residual Connection
        self.residual = nn.Conv2d(1, cnn_out_channels, kernel_size=(num_channels, 1))

        # Adaptive Pooling for d_model alignment
        self.pool = nn.AdaptiveAvgPool2d((None, d_model))

    def forward(self, x):
        """
        Args:
            x: EEG input tensor of shape (batch_size, seq_length, num_channels).
        Returns:
            Processed EEG features: shape (batch_size, seq_length, d_model).
        """
        x = x.permute(0, 2, 1).unsqueeze(1)  # Reshape to (batch_size, 1, num_channels, seq_length)
        
        residual = self.residual(x)  # Residual connection
        cnn_features = self.cnn(x)
        temporal_features = self.temporal_conv(cnn_features)  # Temporal convolution
        cnn_features = cnn_features + temporal_features  # Combine features
        cnn_features += residual  # Add residual
        
        cnn_features = cnn_features.squeeze(2)  # Remove channel dimension
        final_features = self.pool(cnn_features)  # Align to d_model
        
        return final_features
