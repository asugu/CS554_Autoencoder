import torch
import torch.nn as nn

# We did try using skip connections only inside the encoder but they did not improve the quality
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class BaseCAE(nn.Module):
    def __init__(self, color_channel, flatten_size, hidden_dim, latent_dim, tupled_flatten) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(color_channel, 16, 3, stride = 2 , padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_linear = nn.Sequential(
            nn.Linear(flatten_size, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, flatten_size)
        )
        self.unflatten = nn.Unflatten(dim = 1, unflattened_size = tupled_flatten)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,output_padding=1), ## Add output_padding = 1 for CIFAR10, remove for MNIST
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16 , 3, stride=2, padding=1,output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, color_channel, 3, stride=2, padding=1,output_padding=1)
        )
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)

        # Decoder
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x
    
    
    
    
class ImprovedCAE(nn.Module):
    def __init__(self, color_channel, flatten_size, hidden_dim, latent_dim, tupled_flatten) -> None:
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(color_channel, 16, 3, stride = 2 , padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        self.enc4 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_linear = nn.Sequential(
            nn.Linear(flatten_size, latent_dim),
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, flatten_size),
        )
        self.unflatten = nn.Unflatten(dim = 1, unflattened_size = tupled_flatten)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  ## Add output_padding = 1 for CIFAR10, remove for MNIST
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16 , 3, stride=2, padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(16, color_channel, 3, stride=2, padding=1,output_padding=1)
        )
        
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x = self.flatten(x4)
        x = self.encoder_linear(x)

        # Decoder
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = x4 + x
        x = self.dec1(x)
        x = x3 + x
        x = self.dec2(x)
        x + x2 + x
        x = self.dec3(x)
        x = x1 + x
        x = self.dec4(x)
        x = torch.sigmoid(x)
        return x
    
    

