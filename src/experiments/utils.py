import torch
import numpy as np
import time
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.fid import FrechetInceptionDistance
from ddpm.utils import save_pilimg, pilimg_to_tensor
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load a pre-trained classifier (for classifier accuracy)
fid_metric = FrechetInceptionDistance(feature=2048)

# Function to compute PSNR
def compute_psnr(img_true, img_recon):
    img_true = img_true.cpu().numpy()
    img_recon = img_recon.cpu().numpy()
    psnr_value = peak_signal_noise_ratio(img_true, img_recon, data_range=1.0)
    return psnr_value

# Function to compute SSIM
def compute_ssim(img_true, img_recon):
    img_true = img_true.cpu().numpy().transpose(1, 2, 0)
    img_recon = img_recon.cpu().numpy().transpose(1, 2, 0)
    ssim_value = structural_similarity(img_true, img_recon, data_range=1.0, channel_axis=-1)
    return ssim_value

# Function to compute FID
def compute_fid(real_images, generated_images):
    # Convert images from float (0,1) to uint8 (0,255)
    real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
    generated_images = (generated_images * 255).clamp(0, 255).to(torch.uint8)

    fid_metric.update(real_images, real=True)
    fid_metric.update(generated_images, real=False)
    return fid_metric.compute().item()

# Function to measure inference speed
def measure_inference_speed(model, measurement_operator, test_loader, device="cuda"):
    model.to(device)
    times = []
    
    for batch in test_loader:
        real_images, _ = batch
        real_images = real_images.to(device)
        measurement = measurement_operator(real_images)  # Apply the measurement operator
        
        start_time = time.time()
        reconstructed_images = model.sample(measurement)
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time_per_image = np.mean(times)
    return avg_time_per_image


def evaluate_model(sampler, test_loader, num_steps = 100, sigma_y=None, noiseless=True, seed=None, device="cuda"):
    psnr_list, ssim_list, fid_real, fid_fake = [], [], [], []

    real_images_list, recon_images_list = [], []
    
    for batch in test_loader:
        real_images = batch.to(device)
        measurement = sampler.measurement_operator(real_images)

        # Generate reconstructed images
        reconstructed_images = sampler.sample(measurement, num_steps, sigma_y, noiseless, seed)

        for i in range(real_images.size(0)):
            psnr_list.append(compute_psnr(real_images[i], reconstructed_images[i]))
            ssim_list.append(compute_ssim(real_images[i], reconstructed_images[i]))

        real_images_list.append(real_images.cpu())
        recon_images_list.append(reconstructed_images.cpu())

    # Compute FID
    fid_real = torch.cat(real_images_list)
    fid_fake = torch.cat(recon_images_list)
    fid_score = compute_fid(fid_real, fid_fake)

    # Compute final averages
    results = {
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "FID": fid_score,
        "all_PSNR": psnr_list,
        "all_SSIM": ssim_list,
    }
    
    return results


def save_results(result, original_img, naive_image, path):
    out = torch.cat((original_img, result, naive_image), dim = 2)
    save_pilimg(out, path)
    

# def plot_results(results, save_path="results/super_resolution/"):
#     # Ensure the save directory exists
#     os.makedirs(save_path, exist_ok=True)

#     plt.figure(figsize=(12, 5))

#     # Histogram of PSNR values
#     plt.subplot(1, 2, 1)
#     plt.hist(results["all_PSNR"], bins=20, color='blue', alpha=0.7, edgecolor='black')
#     plt.axvline(results["PSNR"], color='red', linestyle='dashed', linewidth=2, label=f'Avg PSNR: {results["PSNR"]:.2f}')
#     plt.xlabel("PSNR (dB)")
#     plt.ylabel("Frequency")
#     plt.title("Distribution of PSNR Scores")
#     plt.legend()

#     # Histogram of SSIM values
#     plt.subplot(1, 2, 2)
#     plt.hist(results["all_SSIM"], bins=20, color='green', alpha=0.7, edgecolor='black')
#     plt.axvline(results["SSIM"], color='red', linestyle='dashed', linewidth=2, label=f'Avg SSIM: {results["SSIM"]:.2f}')
#     plt.xlabel("SSIM")
#     plt.ylabel("Frequency")
#     plt.title("Distribution of SSIM Scores")
#     plt.legend()

#     # Save Accuracy plot
#     plt.savefig(os.path.join(save_path, "psnr_ssim.png"))

#     plt.tight_layout()
#     plt.show()


def plot_results(results, save_path="results/super_resolution/"):
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    
    # Histogram of PSNR values
    axes[0].hist(results["all_PSNR"], bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(results["PSNR"], color='red', linestyle='dashed', linewidth=2, label=f'Avg PSNR: {results["PSNR"]:.2f}')
    axes[0].set_xlabel("PSNR (dB)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Distribution of PSNR Scores", fontsize=14)
    axes[0].legend()

    # Histogram of SSIM values
    axes[1].hist(results["all_SSIM"], bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(results["SSIM"], color='red', linestyle='dashed', linewidth=2, label=f'Avg SSIM: {results["SSIM"]:.2f}')
    axes[1].set_xlabel("SSIM", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Distribution of SSIM Scores", fontsize=14)
    axes[1].legend()

    # Adjust spacing
    plt.tight_layout()  
    plt.subplots_adjust(wspace=0.3)  

    # Save the figure
    save_file = os.path.join(save_path, "psnr_ssim.png")
    plt.savefig(save_file, bbox_inches='tight', dpi=300)  

    # Show the plot
    plt.show()


class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  
        
        if self.transform:
            image = self.transform(image)
        
        return pilimg_to_tensor(image).squeeze(0)
