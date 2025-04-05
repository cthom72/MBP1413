import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.transforms import (
    RandGaussianNoised,
    RandRicianNoised,
)
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pandas as pd

# set rcparams for font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18

# load the image
def load_image(filename, resize_dim=(180, 180)):
    """
    Load a grayscale image, resize to resize_dim, normalize to [0, 1],
    and convert to a torch tensor of shape (1, H, W).
    """

    # open the image
    img = Image.open(filename).convert('L')

    # resize
    img = img.resize(resize_dim, Image.BILINEAR)

    # normalize
    img_np = np.array(img).astype(np.float32) / 255.0

    # Add channel dimension
    img_np = np.expand_dims(img_np, axis=0)
    return torch.from_numpy(img_np)

def apply_noise(image, noise_type, noise_params, idx):
    """
    Apply the specified noise type to a single image tensor.
    The image tensor is assumed to be of shape (1, H, W).
    Returns the noisy image and the noise strength used.
    """
    if noise_type == "gaussian":
        std_min = noise_params.get("std_min", 10)
        std_max = noise_params.get("std_max", 30)
        noise_std = np.random.uniform(std_min, std_max) / 255
        noise_mean = noise_params.get("mean", 0.0)
        noise_transform = RandGaussianNoised(
            keys=["image"],
            prob=1.0,
            mean=noise_mean,
            std=noise_std
        )
        data = {"image": image}
        noisy_data = noise_transform(data)
        noisy_image = torch.clamp(noisy_data["image"], 0, 1)
        noise_strength = noise_std

    elif noise_type == "rician":
        std_min = noise_params.get("std_min", 10)
        std_max = noise_params.get("std_max", 30)
        noise_std = np.random.uniform(std_min, std_max) / 255
        noise_transform = RandRicianNoised(
            keys=["image"],
            prob=1.0,
            std=noise_std
        )
        data = {"image": image}
        noisy_data = noise_transform(data)
        noisy_image = torch.clamp(noisy_data["image"], 0, 1)
        noise_strength = noise_std

   # type
    elif noise_type == "gibbs":

        # minimum
        alpha_min = noise_params.get("alpha_min", 0.05)

        # maximum
        alpha_max = noise_params.get("alpha_max", 0.5)

        # sample
        alpha = np.random.uniform(alpha_min, alpha_max)

        # squeeze
        img_np = image.squeeze().cpu().numpy()

        # shape
        H, W = img_np.shape

        # fft
        kspace = np.fft.fft2(img_np)

        # shift
        kspace_shifted = np.fft.fftshift(kspace)

        # radius
        max_radius = min(H, W) / 2.0

        # filter
        radius = (1.0 - alpha) * max_radius

        # grid
        yy, xx = np.ogrid[:H, :W]

        # center
        cy, cx = H // 2, W // 2

        # square
        dist_sq = (yy - cy)**2 + (xx - cx)**2

        # root
        dist = np.sqrt(dist_sq)

        # falloff
        falloff = 1.0 / (1.0 + np.exp((dist - radius) / (alpha * 10)))

        # sample
        noise_mask = np.random.normal(loc=falloff, scale=0.05)

        # clip
        noise_mask = np.clip(noise_mask, 0.0, 1.0)

        # apply k space mask
        kspace_shifted_masked = kspace_shifted * noise_mask

        # unshift
        kspace_unshifted = np.fft.ifftshift(kspace_shifted_masked)

        # inverse ft
        img_gibbs = np.abs(np.fft.ifft2(kspace_unshifted))

        # clip
        img_gibbs = np.clip(img_gibbs, 0, 1)

        # convert to tensor
        noisy_image = torch.tensor(img_gibbs, dtype=image.dtype, device=image.device).unsqueeze(0)

        # output
        noise_strength = alpha


    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    return noisy_image, noise_strength


def main():
    
    # file name
    filename = "/cluster/home/t134723uhn/LowFieldMRI/data_v2/split2/IXI017_Guys_060.png" 
    image = load_image(filename, resize_dim=(180, 180))
    
    # noise params
    noise_params_dict = {
        "gaussian": {"std_min": 10, "std_max": 30, "mean": 0.0},
        "rician": {"std_min": 10, "std_max": 30},
        "gibbs": {"alpha_min": 0.1, "alpha_max": 1.0},
    }
    
    noise_types = ["gaussian", "rician", "gibbs"]

    # number of iters to consider
    iterations = 500
    psnr_results = {n: [] for n in noise_types}
    ssim_results = {n: [] for n in noise_types}
    
    # get the clean image as a numpy array for metric computation.
    clean_np = image.squeeze().cpu().numpy()
    
    # loop over each noise type and accumulate PSNR and SSIM values.
    for noise in noise_types:
        params = noise_params_dict[noise]
        for i in range(iterations):
            noisy_image, strength = apply_noise(image, noise, params, i)
            noisy_np = noisy_image.squeeze().cpu().numpy()
            current_psnr = psnr(clean_np, noisy_np, data_range=1.0)
            current_ssim = ssim(clean_np, noisy_np, data_range=1.0)
            psnr_results[noise].append(current_psnr)
            ssim_results[noise].append(current_ssim)
        psnr_arr = np.array(psnr_results[noise])
        ssim_arr = np.array(ssim_results[noise])

        # PRINT ALL THE RESULTS
        print(f"Noise type: {noise}")
        print(f"Average PSNR: {psnr_arr.mean():.2f} dB, Std PSNR: {psnr_arr.std():.2f} dB")
        print(f"Average SSIM: {ssim_arr.mean():.4f}, Std SSIM: {ssim_arr.std():.4f}\n")
    
if __name__ == "__main__":
    main()

