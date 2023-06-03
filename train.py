import os
import yaml
import torch
import logging

import torch.nn as nn
import numpy as np

from dataset import *
from networks import *
from utils import plot_ae_outputs, plot_loss, plot_ae_outputs_cinic
from losses import CombinedLoss


config_file = 'config.yaml'


with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

save_location = config['save_location']
if not os.path.exists("results"):
    os.makedirs("results")

if not os.path.exists("results/" + save_location):
    os.makedirs("results/" +save_location)
    
logging.basicConfig(filename = "results/" +save_location+'/log.log',
                    filemode = 'w',
                    level = logging.DEBUG,
                    format=""
                    )
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('PIL').setLevel(logging.WARNING)

if config['dataset'] == 'MNISTDataset':
    ds = MNISTDataset(root = './mnist')
    train = ds.train
    val = ds.val
    test = ds.test
    logging.info("Selected Dataset: MNIST")
elif config['dataset'] == 'CIFAR10Dataset':
    ds = CIFAR10Dataset(root = './cifar10')
    train = ds.train
    val = ds.val
    test = ds.test
    logging.info("Selected Dataset: CIFAR10")
elif config['dataset'] == 'CINIC10Dataset':
    ds = CINIC10Dataset(root = './cinic10')
    train = ds.train
    val = ds.val
    test = ds.test
    logging.info("Selected Dataset: CINIC10")


batch_size = config['batch_size']
lr = config['lr']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
logging.info(f"Selected device: {device}")
logging.info(f"Batch Size: {batch_size}")

# flatten pre_process
flatten_size = config['flatten_size']
dimensions = [int(dim.strip()) for dim in flatten_size.split('*')]
tupled_flatten = tuple(dimensions)


if config['network'] == 'BaseCAE':
    model = BaseCAE(color_channel=config['color_channel'], flatten_size=eval(flatten_size), hidden_dim=config['hidden_dim'],
                      latent_dim=config['latent_dim'], tupled_flatten=tupled_flatten)
    model.to(device)
    logging.info("Selected network: BaseCAE")
    logging.info(f"Network flatten_size: {tupled_flatten} \nLatent Dim: {config['latent_dim']} \nHidden Dim: {config['hidden_dim']}")
    
## Add elif statements for additional networks
elif config['network'] == 'ImprovedCAE':
    model = ImprovedCAE(color_channel=config['color_channel'], flatten_size=eval(flatten_size), hidden_dim=config['hidden_dim'],
                      latent_dim=config['latent_dim'], tupled_flatten=tupled_flatten)
    model.to(device)
    logging.info("Selected network: ImprovedCAE")
    logging.info(f"Network flatten_size: {tupled_flatten} \nLatent Dim: {config['latent_dim']} \nHidden Dim: {config['hidden_dim']}")


if config['loss'] == 'MSE':
    criterion = nn.MSELoss()
    logging.info('Selected loss function: MSE')
elif config['loss'] == 'L1':
    criterion = nn.L1Loss()
    logging.info('Selected loss L1')
elif config['loss'] == 'BCE':
    criterion = nn.BCELoss()
    logging.info('Selected loss BCE')
elif config['loss'] == 'Combined':
    weight_mse, weight_l1 = 1.0, 0.5 ## give weights manually here 
    criterion = CombinedLoss(weight_mse=weight_mse, weight_l1=weight_l1) 
    logging.info(f"Selected loss combined with W_MSE {weight_mse} W_L1{weight_l1}")
    
if config['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr, weight_decay=1e-5)
    logging.info("Selected Optimizer: Adam")
    logging.info(f"Learning Rate: {lr}")

elif config['optimizer'] == 'RMS':
    optimizer = torch.optim.RMSprop(params=model.parameters(),lr=lr, weight_decay=1e-5)
    logging.info("Selected Optimizer: RMSProp")
    logging.info(f"Learning Rate: {lr}")

if config['optimizer'] == 'SGD':
    optimizer = torch.optim.SGD(params=model.parameters(),lr=lr, weight_decay=1e-5)
    logging.info("Selected Optimizer: SGD")
    logging.info(f"Learning Rate: {lr}")

## Add elif statements for additional optimizers

EPOCH = config['epoch']
print("Training Started...")
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
        for step, (batch, _) in enumerate(val_loader):
            batch = batch.to(device)
            output = model(batch)
            preds.append(output)
            ys.append(batch)
        preds, ys = torch.cat(preds), torch.cat(ys)
        val_loss = criterion(preds, ys)
        validation_loss.append(val_loss.data)
        
    logging.info(
         f"Epoch: {epoch}, Train Loss: {mean_train_loss[epoch]:.4f}, Val Loss: {validation_loss[epoch]:.4f}"
    )
if config['dataset'] != "CINIC10Dataset":
     avg_psnr, avg_ssim = plot_ae_outputs(model, test, "./results/" + save_location)
else:
    avg_psnr, avg_ssim = plot_ae_outputs_cinic(model, test, "./results/" + save_location)
logging.info(f"Average PSNR: {avg_psnr:.2f}")
logging.info(f"Average SSIM: {avg_ssim:.4f}")
plot_loss(mean_train_loss, validation_loss, np.linspace(0, EPOCH, EPOCH), "./results/" + save_location)
print("Training Ended...")