import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from Autoencoder import EEGAutoencoder
from train_AE import train_epoch_autoencoder

# Example EEG dataset (replace with your actual data)
def create_example_dataset(num_samples=1000, seq_len=300, num_channels=14):
    data = torch.rand(num_samples, seq_len, num_channels)  # Random EEG-like data
    print(data.shape)
    return TensorDataset(data)

'''
Example of working, to remove in the concrete implementation. Attach to the transformer architecture only the part of
'Autoencoder.py' should be a good skeleton for the needs. I leave to the great @Vincenzo and @Renato the integration in the
beatiful code already avaible.
'''
def main():
    # Parameters
    torch.cuda.empty_cache()
    input_dim = 14
    max_seq_len = 50000
    latent_dim = 128  # Reduced latent dimension
    batch_size = 2
    num_epochs = 20
    chunk_size = 1000
    learning_rate = 0.001

    # Dataset and DataLoader
    dataset = create_example_dataset(100,max_seq_len,input_dim)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,pin_memory=True)

    # Model
    model = EEGAutoencoder(input_dim, chunk_size, latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting the training")
    # Train the model
    trained_model = train_epoch_autoencoder(1, data_loader, model, criterion_loss, optimizer, device)

    # Monitor GPU memory
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

if __name__ == "__main__":
    main()
