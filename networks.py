import torch
import torch.nn as nn

class BaseCAE(nn.Module):
    def __init__(self, color_channel, flatten_size, hidden_dim, latent_dim, tupled_flatten) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(color_channel, 32, 3, stride = 2 , padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1),
            nn.ReLU(True)
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
            nn.ConvTranspose2d(64, 32 , 3, stride=2, padding=1,output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, color_channel, 3, stride=2, padding=1,output_padding=1)
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
    
        