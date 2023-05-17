import numpy as np
import torch
import torch.nn as nn


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = []
  
    for image_batch, _ in dataloader:        
        image_batch = image_batch.to(device)
       
        decoded_data = model(image_batch)
      
        loss = loss_fn(decoded_data, image_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test_epoch(model, device, dataloader, loss_fn):
    model.eval()
    
    with torch.no_grad(): 
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            
            decoded_data = model(image_batch)
            
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

class AE_lin(nn.Module):    # For MNIST
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)
 
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        #x = self.flatten(x)
        #x = self.encoder_lin(x)
        #x = self.decoder_lin(x)
        #x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
class Autoencoder(nn.Module):   # for CİFAR10
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x