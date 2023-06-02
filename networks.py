import torch
import torch.nn as nn

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
    def __init__(self, color_channel) -> None:
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
        )
       
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), ## Add output_padding = 1 for CIFAR10, remove for MNIST
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16 , 3, stride=2, padding=1,output_padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(16),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(16, color_channel, 3, stride=2, padding=1,output_padding=1)
        )
        self.residual_block1 = ResidualBlock(16)
        self.residual_block2 = ResidualBlock(32)
        self.residual_block3 = ResidualBlock(64)
        
        self.residual_block4 = ResidualBlock(32)
        self.residual_block5 = ResidualBlock(16)
        self.residual_block6 = ResidualBlock(color_channel)
        
    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        # x = self.residual_block1(x)
        x = self.enc2(x)
        # x = self.residual_block2(x)
        x = self.enc3(x)
        # x = self.residual_block3(x)


        # Decoder
        x = self.dec1(x)
        # x = self.residual_block4(x)
        x = self.dec2(x)
        # x = self.residual_block5(x)
        x = self.dec3(x)
        # x = self.residual_block6(x)
        x = torch.sigmoid(x)
        return x