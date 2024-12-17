'''import torch
import torch.nn as nn
 
class EEG_cnn(nn.Module):
    def __init__(self, input_channels, output_channels, band_dim):
        super(EEG_cnn, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,  # Numero iniziale di canali (14)
            out_channels=64,  # Primo incremento del numero di canali
            kernel_size=(1, band_dim),  # Kernel che copre completamente la dimensione delle bande
            stride=(1, 1),
            padding=(0, 0)
        )
        self.relu1 = nn.ReLU()  # Attivazione non lineare
        self.batch_norm1 = nn.BatchNorm2d(64)  # Normalizzazione batch
 
        self.conv2 = nn.Conv2d(
            in_channels=64,  # Numero di canali dopo il primo strato
            out_channels=output_channels,  # Numero finale di canali (128)
            kernel_size=(1,3),  # Kernel per catturare la relazione temporale
            stride=(1, 1),
            padding=(0, 1)  # Padding per mantenere la lunghezza della sequenza invariata
        )
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm2d(output_channels)
 
    def forward(self, x):
        # x ha forma (batch_size, input_channels, seq_length, band_dim)
        x = self.conv1(x)  # Dopo la prima convoluzione
        x = self.relu1(x)
        x = self.batch_norm1(x)
 
        x = self.conv2(x)  # Dopo la seconda convoluzione
        x = self.relu2(x)
        x = self.batch_norm2(x)
 
        x = x.squeeze(-1)  # Rimuove la dimensione 1 finale: (batch_size, output_channels, seq_length)
        return x'''
 
'''# Esempio di utilizzo
def main():
    batch_size = 32
    seq_length = 50
    input_channels = 14
    band_dim = 10
    output_channels = 128
 
    # Input tensor con dimensioni (batch_size, input_channels, seq_length, band_dim)
    input_tensor = torch.randn(batch_size, input_channels, seq_length, band_dim)
 
    # Istanza del modello
    model = ProcessTensor(input_channels=input_channels, output_channels=output_channels, band_dim=band_dim)
 
    # Output tensor
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
 
if __name__ == "__main__":
    main()'''
    
import torch
import torch.nn as nn

class EEG_cnn(nn.Module):
    def __init__(self, num_channels, num_band, output_features=128):
        super(EEG_cnn, self).__init__()
        
        # Convoluzione iniziale: riduce num_channels
        self.conv1 = nn.Conv2d(
            in_channels=num_band, 
            out_channels=32, 
            kernel_size=(num_channels, 1),  # Riduce completamente num_channels
            stride=1
        )
        
        # Seconda convoluzione: mantiene W e riduce le feature
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(1, 3), 
            stride=1, 
            padding=(0, 1)  # Padding solo per preservare W
        )
        
        # Terza convoluzione: porta a output_features
        self.conv3 = nn.Conv2d(
            in_channels=64, 
            out_channels=output_features, 
            kernel_size=(1, 3), 
            stride=1, 
            padding=(0, 1)  # Padding solo per preservare W
        )
        
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(output_features)

    def forward(self, x):
        # Input è [B, num_channels, W, num_band]
        x = x.permute(0, 3, 1, 2)  # Cambia in [B, num_band, num_channels, W]
        
        x = self.conv1(x)  # Riduce num_channels a 1
        x = self.relu(x)
        
        x = self.conv2(x)  # Mantiene W
        x = self.relu(x)
        
        x = self.conv3(x)  # Mantiene W e riduce a output_features
        x = self.batch_norm(x)
        x = self.relu(x)
        
        x = x.squeeze(2)  # Rimuove la dimensione ridotta di num_channels
        # Ora x è [B, 128, W]
        return x


