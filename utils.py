
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
config_file = 'config.yaml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

def plot_loss(train_loss_list, val_loss_list, epoch, save_dir):
    save_dir = Path(save_dir)
    plt.figure(figsize=(12, 8))
    plt.plot(epoch, train_loss_list, label="Train Loss", color="red")
    plt.plot(epoch, val_loss_list, label="Validation Loss", color="blue")
    plt.title("Loss")
    plt.legend()
    plt.savefig(save_dir / "loss.png")
    plt.show()

def calculate_metrics(original, reconstructed):
    win_size = 7 if config['color_channel'] == 3 else 11
    channel_axis = 2 if config['color_channel'] == 3 else None
    psnr = peak_signal_noise_ratio(original, reconstructed)  # Determine the appropriate win_size
    ssim = structural_similarity(
        original,
        reconstructed,
        win_size = win_size,
        channel_axis = channel_axis,
        data_range=(original.max() - original.min()),
    )
    return psnr, ssim

def plot_ae_outputs(model, test_dataset, save_dir, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = np.array(test_dataset.dataset.targets)[test_dataset.indices]
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}

    psnr_list = []
    ssim_list = []

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        model.eval()

        with torch.no_grad():
            rec_img = model(img)

        if config['color_channel'] == 1:
            img = img.cpu().squeeze().numpy()
            rec_img = rec_img.cpu().squeeze().numpy()
        else:
            img = img.cpu().squeeze().numpy().transpose(1, 2, 0)
            rec_img = rec_img.cpu().squeeze().numpy().transpose(1, 2, 0)

        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title("Original images")

        ax = plt.subplot(2, n, i + 1 + n)
        rec_img = np.clip(rec_img, 0, 1)
        rec_img = np.clip(rec_img, 0, 255)
        plt.imshow(rec_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n // 2:
            ax.set_title("Reconstructed images")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_dir = Path(save_dir)
        save_path = save_dir / f"reconstructed_{i}.png"
        plt.imsave(save_path, rec_img)

        # Calculate metrics
        psnr, ssim = calculate_metrics(img, rec_img)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    plt.savefig(save_dir / "plot.png")
    plt.close()

    print(f"Average PSNR: {np.mean(psnr_list):.2f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

    return np.mean(psnr_list), np.mean(ssim_list)