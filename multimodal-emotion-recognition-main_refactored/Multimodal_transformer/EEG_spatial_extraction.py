import torch
import torch.nn as nn

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, num_channels, num_bands, output_dim=128):
        super(SpatialFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))  # 32 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        print(64 * num_channels * num_bands // (4**2))
        self.fc = nn.Linear(256, output_dim)  # Flatten + Fully Connected
        
    def forward(self, x):
        # x: [batch_size, 1, num_channels, num_bands]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = nn.functional.adaptive_avg_pool2d(x, (2, 2))  # Optional, reduce spatial dims
        
        x = x.flatten(start_dim=1)  # Flatten to [batch_size, features]
        
  
        x = self.fc(x)  # Map to output_dim
        return x
