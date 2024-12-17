import torch
import torch.nn as nn


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim=128, num_heads=4, num_layers=2):
        super(TemporalFeatureExtractor, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: [seq_length, batch_size, input_dim]
        x = self.transformer(x)  # Output: [seq_length, batch_size, input_dim]
        return x
