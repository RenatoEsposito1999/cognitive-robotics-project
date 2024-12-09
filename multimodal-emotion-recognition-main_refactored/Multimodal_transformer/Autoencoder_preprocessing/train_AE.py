import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

def train_epoch_autoencoder(epoch, data_loader_eeg, model, criterion_loss, optimizer,device):
    scaler = GradScaler()
    num_epochs = 20 # To remove
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in data_loader_eeg:
            # Move data to the device
            eeg_data = batch[0].to(device)  # Shape: [batch_size, max_seq_len, input_dim]
            # Forward pass
            latent, reconstructed = model(eeg_data)

            # Compute loss
            loss = criterion_loss(reconstructed, eeg_data)
            total_loss += loss.item()

            torch.cuda.empty_cache()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader_eeg)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model
