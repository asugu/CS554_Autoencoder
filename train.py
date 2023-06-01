
import yaml
import torch

import torch.nn as nn
import numpy as np

from dataset import *
from networks import *
from utils import plot_ae_outputs, plot_loss


# Specify the path to your YAML file
config_file = 'config.yaml'

# Read the configuration from the YAML file
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

if config['dataset'] == 'MNISTDataset':
    ds = MNISTDataset(root = './mnist')
    train = ds.train
    test = ds.test
    print("Selected Dataset: MNIST")
elif config['dataset'] == 'CIFAR10Dataset':
    ds = CIFAR10Dataset(root = './cifar10')
    train = ds.train
    test = ds.test
    print("Selected Dataset: CIFAR10")

    
batch_size = config['batch_size']
lr = config['lr']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
print(f"Selected device: {device}")
print(f"Batch Size: {batch_size}")
print()

# flatten pre_process
flatten_size = config['flatten_size']
dimensions = [int(dim.strip()) for dim in flatten_size.split('*')]
tupled_flatten = tuple(dimensions)


if config['network'] == 'BaseCAE':
    model = BaseCAE(color_channel=config['color_channel'], flatten_size=eval(flatten_size), hidden_dim=config['hidden_dim'],
                      latent_dim=config['latent_dim'], tupled_flatten=tupled_flatten)
    model.to(device)
    print("Selected network: BaseCAE")
    print(f"Network flatten_size: {tupled_flatten} \nLatent Dim: {config['latent_dim']} \nHidden Dim: {config['hidden_dim']}")
    print()
## Add elif statements for additional networks


if config['loss'] == 'MSE':
    criterion = nn.MSELoss()
    print('Selected loss function: MSE')
    
### Add elif statements for additional losses

if config['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr, weight_decay=1e-5)
    print("Selected Optimizer: Adam")
    print(f"Learning Rate: {lr}")

## Add elif statements for additional optimizers

EPOCH = config['epoch']

mean_train_loss, validation_loss = [], []
for epoch in range(EPOCH):
    train_loss = []
    model.train()
    for step, (batch, _) in enumerate(train_loader):
        batch = batch.to(device)
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    mean_train_loss.append(np.mean(train_loss))

    # Validation
    model.eval()
    with torch.no_grad():
        preds = []
        ys = []
        for step, (batch, _) in enumerate(test_loader):
            batch = batch.to(device)
            output = model(batch)
            preds.append(output.cpu())
            ys.append(batch.cpu())
        preds, ys = torch.cat(preds), torch.cat(ys)
        val_loss = criterion(preds, ys)
        validation_loss.append(val_loss.data)

    print(
        f"Epoch: {epoch}, Train Loss: {mean_train_loss[epoch]:.4f}, Val Loss: {validation_loss[epoch]:.4f}"
    )
plot_ae_outputs(model, test, "./mnist_saved")
plot_loss(mean_train_loss, validation_loss, np.linspace(0, EPOCH, EPOCH), './mnist_saved')