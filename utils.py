from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch

def ssim(img1, img2, sigma=1.5, L=255):
    k1 = 0.01
    k2 = 0.03
    window = create_gaussian_window(img1.shape[0], img1.shape[1], sigma)

    mu1 = apply_window(gaussian_filter(img1, sigma), window)
    mu2 = apply_window(gaussian_filter(img2, sigma), window)

    sigma1_sq = apply_window(gaussian_filter(img1**2, sigma), window) - mu1**2
    sigma2_sq = apply_window(gaussian_filter(img2**2, sigma), window) - mu2**2
    sigma12 = apply_window(gaussian_filter(img1 * img2, sigma), window) - mu1 * mu2

    C1 = (k1 * L)**2
    C2 = (k2 * L)**2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

def create_gaussian_window(width, height, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    d = np.sqrt(x * x + y * y)
    window = np.exp(-((d ** 2) / (2.0 * sigma ** 2)))
    return window

def apply_window(img, window):
    if len(img.shape) == 3:
        return np.mean(img * window[:, :, np.newaxis], axis=(0, 1))
    else:
        return np.mean(img * window)
    
def psnr(img1, img2, max_value=255):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_value ** 2) / mse)

def plot_ae_outputs(model, diz_loss,device,test_dataset,test_targets , test_indices, key, n=10):
    fig, axs = plt.subplots(2, n, figsize=(16, 7))
    targets = test_targets[test_indices].numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    clear_output(wait=True)

    psnr_values = []
    ssim_values = []

    for i in range(n):
        img_ax = axs[0, i]
        rec_ax = axs[1, i]

        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            rec_img = model(img)

        if key == 'MNIST':
            img_np = img.cpu().squeeze().numpy()
            rec_img_np = rec_img.cpu().squeeze().numpy()
        elif key == 'CIFAR10':
            img_np = img.cpu().squeeze().numpy().transpose(1, 2, 0)  
            rec_img_np = rec_img.cpu().squeeze().numpy().transpose(1, 2, 0)    

        img_ax.imshow(img_np, cmap='gist_gray')
        img_ax.get_xaxis().set_visible(False)
        img_ax.get_yaxis().set_visible(False)
        if i == n // 2:
            img_ax.set_title('Original images')

        rec_ax.imshow(rec_img_np ,cmap='gist_gray')
        rec_ax.get_xaxis().set_visible(False)
        rec_ax.get_yaxis().set_visible(False)
        if i == n // 2:
            rec_ax.set_title('Reconstructed images')

    for i in range(len(test_dataset)):
        img = test_dataset[i][0].unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            rec_img = model(img)

        if key == 'MNIST':
            img_np = img.cpu().squeeze().numpy()
            rec_img_np = rec_img.cpu().squeeze().numpy()
        elif key == 'CIFAR10':
            img_np = img.cpu().squeeze().numpy().transpose(1, 2, 0)  
            rec_img_np = rec_img.cpu().squeeze().numpy().transpose(1, 2, 0)   

        psnr_value = psnr(img_np, rec_img_np)
        ssim_value = ssim(img_np, rec_img_np)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
    
    fig.tight_layout()

    fig_combined, (ax_loss, ax_psnr, ax_ssim) = plt.subplots(1,3, figsize=(20, 6)) 

    ax_loss.semilogy(diz_loss['train_loss'], label='Train ({:.5f})'.format(diz_loss['train_loss'][-1]))
    ax_loss.semilogy(diz_loss['val_loss'], label='Valid ({:.5f})'.format(diz_loss['val_loss'][-1]))
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Average Loss')
    ax_loss.legend()
    

    ax_psnr.hist(psnr_values)
    ax_psnr.set_xlabel('PSNR')
    ax_psnr.set_ylabel('Count')

    ax_ssim.hist(ssim_values)
    ax_ssim.set_xlabel('SSIM')
    ax_ssim.set_ylabel('Count')

    fig_combined.tight_layout()
    plt.show()

    
