import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import shutil
from eeg_dataset import EEGDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



 
'''class EEGAutoencoder(nn.Module):
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
        return latent, reconstructed'''
        
'''
class EEGAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(EEGAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # Riduce a 128 feature
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)  # Ricostruisce input originale
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''


class EEGAutoencoder(nn.Module):
    def __init__(self,num_channels=14, seq_length = 64,num_bands=5, latent_channels=128):
        super(EEGAutoencoder, self).__init__()
        self.input_dim = num_channels
        self.latent_dim = latent_channels
        self.num_bands = num_bands
        self.seq_length = seq_length

        # Encoder and Decoder for smaller chunks
        self.encoder = nn.Sequential(
            nn.Linear(self.seq_length * self.num_bands, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.seq_length * self.num_bands),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, num_channels, seq_length, num_bands = x.size()
        assert num_channels == self.input_dim


        # Process each chunk
        x = x.view(batch_size * num_channels, -1)
        
        latent = self.encoder(x)
        latent = latent.view(batch_size, num_channels, self.latent_dim)
        reconstructed = self.decoder(latent)         
        reconstructed= reconstructed.view(batch_size, num_channels, seq_length, num_bands)

        return latent, reconstructed

        

def save_checkpoint(state,is_best, train):
    if train:
        torch.save(state, 'Checkpoint_autoencoder.pth')
        '''state["state_dict"]=model.module.state_dict()
        torch.save(state, f'Checkpoint_autoencoder_{state["epoch"]}_cpu_.pth')'''
    if is_best:
        shutil.copyfile('Checkpoint_autoencoder.pth','best_autoencoder.pth')
        #shutil.copyfile(f'Checkpoint_autoencoder_{state["epoch"]}_cpu_.pth',f'best_autoencoder_{state["epoch"]}_cpu_.pth')
 
def train_autoencoder(epoch, model, dataloader, optimizer, criterion, num_epochs):
    epoch_loss = 0
    for inputs, _, mask in dataloader:
        inputs = inputs.to("cuda")
        optimizer.zero_grad()
        _, reconstructed = model(inputs)
        
        loss = criterion(reconstructed, inputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #print(f"Train: Epoch [{epoch + 1}/{num_epochs}], Loss data: {loss.item()}")
    if(epoch % 5 == 0):
        print(f"Train: Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    
        
        
def val_autoencoder(epoch, model, dataloader, criterion, num_epochs):
    model.eval()
    epoch_loss = 0
    for inputs, _, mask in dataloader:
        inputs = inputs.to("cuda")
        with torch.no_grad():
            _, reconstructed = model(inputs)
        loss = criterion(reconstructed, inputs)
        epoch_loss += loss.item()
        #print(f"Val: Epoch [{epoch + 1}/{num_epochs}], Loss data: {loss.item()}")
    if (epoch % 5 == 0):
        print(f"Val: Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    return epoch_loss / len(dataloader)


def testing(model, dataloader, criterion):
    model.eval()
    # Load the model's state
    state = torch.load("best_autoencoder.pth")
    model.load_state_dict(state["state_dict"])
    sum_loss = 0
    pca = PCA(n_components=3)  # Change to 3 components
    # Collect all latent representations and labels
    all_latent_data = []
    all_labels = []
    for inputs, labels, mask in dataloader:
        inputs = inputs.to("cuda")
        with torch.no_grad():
            latent_data, reconstruction = model(inputs)
        labels = labels.repeat(128)
        # Reshape latent data to 2D: (num_samples, 14) -> (num_samples, 14)
        latent_data = latent_data.view(-1, 14)
        latent_data = latent_data.to("cpu").numpy()
        all_latent_data.append(latent_data)
        all_labels.append(labels.numpy())
        reconstruction = reconstruction.to("cuda")
        loss = criterion(reconstruction, inputs)
        sum_loss += loss.item()
    # Concatenate all latent data and labels
    all_latent_data = np.concatenate(all_latent_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # Reduce to 3D using PCA
    latent_3D = pca.fit_transform(all_latent_data)
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot with color coding
    scatter = ax.scatter(latent_3D[:, 0], latent_3D[:, 1], latent_3D[:, 2], 
                         c=all_labels, 
                         cmap='viridis', 
                         s=5, 
                         edgecolors='k')
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_zlabel("Latent Dimension 3")
    ax.set_title("3D Latent Space with Color Coding for Labels")
    # Add a color bar
    plt.colorbar(scatter)
    plt.savefig("latent_3d.png", dpi=300)
    plt.show()
    print(f"Loss: {sum_loss / len(dataloader):.4f}")
    
def split_dataset(dataset):
        """
        Splits a dataset into separate datasets based on labels.
    
        Args:
            dataset (Dataset): A dataset object with labeled data.
    
        Returns:
            tuple: Four datasets corresponding to labels 0, 1, 2, and 3.
        """
        # Initialize a dictionary to group indices by labels
        label_dict = {0: [], 1: [], 2: [], 3: []}
    
        # Populate the dictionary with indices corresponding to each label
        for index in range(len(dataset)):
            _, label, mask = dataset[index]
            label_dict[label.item()].append(index)  # Use .item() to get the integer value from the tensor
    
        # Create separate datasets for each label
        dataset_0 = torch.utils.data.Subset(dataset, label_dict[0])
        dataset_1 = torch.utils.data.Subset(dataset, label_dict[1])
        dataset_2 = torch.utils.data.Subset(dataset, label_dict[2])
        dataset_3 = torch.utils.data.Subset(dataset, label_dict[3])
        
        complete_dataset = [dataset_0, dataset_1, dataset_2, dataset_3]
              
        return complete_dataset 
    
def predict(model, split_dt):
    model.eval()
    random_index_happy = torch.randint(0, len(split_dt[1]), (1,)).item()
    sample, labels, mask = split_dt[1][random_index_happy]
    sample = sample.unsqueeze(0)
    
    random_index_angry = torch.randint(0, len(split_dt[1]), (1,)).item()
    sample_angry, labels_angry, mask_angry = split_dt[1][random_index_angry]
    sample_angry = sample_angry.unsqueeze(0)
    sample = sample.to("cuda")
    sample_angry = sample_angry.to("cuda")
    with torch.no_grad():
        latent_happy, _ = model(sample)
        latent_angry, _ = model(sample_angry)
    
    result = torch.sum(torch.abs(latent_happy - latent_angry))
    
    print(result)
    
    

def training_validation(model, train_dataloader, val_dataloader, test_dataloader, num_epochs=300, learning_rate=0.04):
    criterion = nn.MSELoss()  # Mean Squared Error loss
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                dampening=0.9,
                weight_decay=1e-3,
                nesterov=False)
    #scheduler = lr_scheduler.StepLR(optimizer, 20, 0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_acc = float("inf")
    for epoch in range(num_epochs):
        train_autoencoder(epoch, model, train_dataloader, optimizer, criterion, num_epochs)
        
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
            }
                 
        #Save pth
        save_checkpoint(state, False, train=True)
        #scheduler.step()
        
        accuracy = val_autoencoder(epoch, model, val_dataloader, criterion, num_epochs)
        
        if (accuracy < best_acc):
            best_acc = accuracy
            save_checkpoint(None, True, train=False)
            
    testing(model, test_dataloader, criterion)
        

# Example usage
if __name__ == "__main__":
    
    EEGDataset_complete = EEGDataset("/home/v.mele/cognitive_robotics/EEG_data_smooth", 14)
   
    
    total_size_dataset = len(EEGDataset_complete)
    size_training = int(0.7 * total_size_dataset) #70% of complete dataset
    size_validation = int(0.2 * total_size_dataset) #20% of complete dataset
    size_test = total_size_dataset - size_training - size_validation #10% of complete dataset
    
    #Split the complete dataset into three subsets: Training, Validation, Testing
    train_split_eeg, validation_split_eeg, test_split_eeg = torch.utils.data.random_split(EEGDataset_complete, [size_training, size_validation , size_test])
    
    EEG_train_dataloader = DataLoader(train_split_eeg, batch_size=3, shuffle=True, pin_memory=True)
    EEG_val_dataloader = DataLoader(validation_split_eeg, batch_size=3, shuffle=True, pin_memory=True)
    EEG_test_dataloader = DataLoader(test_split_eeg, batch_size=3, shuffle=True, pin_memory=True)
    
    # Model
    model = EEGAutoencoder(num_channels=14, num_bands=5, latent_channels=128)
    model = model.to("cuda")
    #criterion = nn.MSELoss()  # Mean Squared Error loss
    #testing(model, EEG_test_dataloader, criterion)
    #split_dt = split_dataset(test_split_eeg)
    #predict(model, split_dt)
    
    training_validation(model, EEG_train_dataloader, EEG_val_dataloader, EEG_test_dataloader)
    
    