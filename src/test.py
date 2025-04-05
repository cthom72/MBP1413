# test.py

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import create_data_loaders
from models import DnCNN
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import datetime
import pandas as pd

# set fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Verdana', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# TEST.PY

def find_best_model(save_dir):
    """
    Find the best model in the save directory
    
    Args:
        save_dir: Directory containing model files
        
    Returns:
        Path to the best model file or None if no model exists
    """
    # look for best_model.pth
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path
    
    # look for final_model.pth if no best_model.pth
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    if os.path.exists(final_model_path):
        return final_model_path
    
    # look for latest checkpoint if none of above
    checkpoint_files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    if checkpoint_files:
        # extract epoch numbers
        epoch_numbers = []
        for file in checkpoint_files:
            try:
                # extract the epoch number from the filename
                epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
                epoch_numbers.append((epoch, file))
            except:
                continue
        
        # sort by epoch number and return the latest one
        if epoch_numbers:
            epoch_numbers.sort(reverse=True)
            return epoch_numbers[0][1]
    
    return None


def calculate_metrics(original, denoised):
    """
    Calculate PSNR and SSIM metrics
    
    Args:
        original: Original clean image (torch tensor)
        denoised: Denoised image (torch tensor)
        
    Returns:
        Tuple (psnr_value, ssim_value)
    """
    # convert to numpy arrays
    original_np = original.squeeze().cpu().numpy()
    denoised_np = denoised.squeeze().cpu().numpy()
    
    # calculate PSNR
    psnr_value = psnr(original_np, denoised_np, data_range=1.0)
    
    # calculate SSIM
    ssim_value = ssim(original_np, denoised_np, data_range=1.0)
    
    return psnr_value, ssim_value



def apply_fixed_noise(clean, noise_type, param_value, batch_idx, seed):
    """
    Apply fixed, deterministic noise to clean images based on noise type and parameter value
    
    Args:
        clean: Clean image tensor
        noise_type: Type of noise ('gaussian', 'rician', 'gibbs', 'spike')
        param_value: Parameter value for the noise (sigma, alpha, or intensity)
        batch_idx: Batch index for seeding
        seed: Base seed value
        
    Returns:
        Noisy image tensor
    """
    # set seed based on batch index for reproducibility
    base_seed = seed + batch_idx
    
    if noise_type == "gaussian":
        # param_value is sigma (standard deviation)
        # normalize to 0-1 scale
        sigma = param_value / 255.0  
        
        torch.manual_seed(base_seed)
        noise = torch.randn_like(clean) * sigma
        noisy = torch.clamp(clean + noise, 0, 1)
    
    elif noise_type == "rician":
        # param_value is sigma (standard deviation)
        sigma = param_value / 255.0 
        
        torch.manual_seed(base_seed)
        real_noise = torch.randn_like(clean) * sigma
        
        torch.manual_seed(base_seed + 10000)
        imag_noise = torch.randn_like(clean) * sigma
        
        # complex value: clean + real_noise + i*imag_noise
        # take magnitude: |z| = sqrt(real^2 + imag^2)
        noisy = torch.sqrt((clean + real_noise)**2 + imag_noise**2)
        noisy = torch.clamp(noisy, 0, 1)
    
    elif noise_type == "gibbs":
        # param_value is the alpha (0.0 - 1.0), determining strength of high-frequency removal
        alpha = param_value
        batch_size, channels, height, width = clean.shape
        noisy = clean.clone()

        for i in range(batch_size):
            img_np = clean[i, 0].cpu().numpy()

            # FFT and shift to k-space
            kspace = np.fft.fft2(img_np)
            kspace_shifted = np.fft.fftshift(kspace)

            # freate soft radial mask
            cy, cx = height // 2, width // 2
            yy, xx = np.ogrid[:height, :width]
            dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)

            max_radius = min(height, width) / 2.0
            radius = (1.0 - alpha) * max_radius

            # sigmoid falloff (soft circular mask)
            falloff = 1.0 / (1.0 + np.exp((dist - radius) / (alpha * 10 + 1e-8)))
            mask = np.clip(falloff, 0, 1)

            # apply mask to k-space
            kspace_masked = kspace_shifted * mask

            # inverse FFT to image domain
            img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
            img_filtered = np.clip(img_filtered, 0, 1)

            noisy[i, 0] = torch.tensor(img_filtered, dtype=clean.dtype, device=clean.device)

        noisy = torch.clamp(noisy, 0, 1)

    return noisy

def create_metrics_plots(results_df, save_path, noise_type, param_value):
    """
    Create and save box plots for PSNR and SSIM metrics
    
    Args:
        results_df: DataFrame containing PSNR and SSIM values
        save_path: Directory to save plots
        noise_type: Type of noise
        param_value: Parameter value used for the noise
    """
    # Set a modern, clean style and larger font scale
    sns.set_theme(
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    sns.set_context("paper", font_scale=1.4)
    
    # Force Matplotlib to use DejaVu Sans
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    # Get noise type label and parameter label for plots
    noise_label = noise_type.capitalize()
    
    if noise_type in ["gaussian", "rician"]:
        param_label = f"σ={param_value}"
        filename_suffix = f"sigma{param_value}"
    elif noise_type == "gibbs":
        param_label = f"α={param_value:.2f}"
        filename_suffix = f"alpha{param_value:.2f}"
    elif noise_type == "spike":
        param_label = f"i={param_value:.2f}"
        filename_suffix = f"intensity{param_value:.2f}"
    else:
        param_label = f"Param={param_value}"
        filename_suffix = f"param{param_value}"

    # PSNR DataFrame
    psnr_df = pd.DataFrame({
        'Metric': results_df['psnr_noisy'],
        'Type': ['Noisy'] * len(results_df)
    })
    psnr_df = pd.concat([psnr_df,
                         pd.DataFrame({
                             'Metric': results_df['psnr_denoised'],
                             'Type': ['Denoised'] * len(results_df)
                         })])

    # SSIM DataFrame
    ssim_df = pd.DataFrame({
        'Metric': results_df['ssim_noisy'],
        'Type': ['Noisy'] * len(results_df)
    })
    ssim_df = pd.concat([ssim_df,
                         pd.DataFrame({
                             'Metric': results_df['ssim_denoised'],
                             'Type': ['Denoised'] * len(results_df)
                         })])
    
    # PSNR BOX PLOT
    plt.figure(figsize=(6, 8))  
    sns.boxplot(
        x='Type',
        y='Metric',
        data=psnr_df,
        width=0.3,         
        showfliers=False,    
        linewidth=1.2,        
        palette=['lightgray', 'skyblue']
    )
    
    plt.title(f'PSNR Before and After Denoising ({noise_label}, {param_label})', pad=15)
    plt.ylabel('PSNR (dB)')
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'psnr_comparison_{filename_suffix}.png'), dpi=400, bbox_inches='tight')
    plt.close()
    
    #SSIM BOX PLOT
    plt.figure(figsize=(6, 8))  
    sns.boxplot(
        x='Type',
        y='Metric',
        data=ssim_df,
        width=0.3,
        showfliers=False,
        linewidth=1.2,
        palette=['lightgray', 'lightgreen']
    )
    
    plt.title(f'SSIM Before and After Denoising ({noise_label}, {param_label})', pad=15)
    plt.ylabel('SSIM')
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'ssim_comparison_{filename_suffix}.png'), dpi=400, bbox_inches='tight')
    plt.close()
    
    # PSNR + SSIM BOX PLOT
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # PSNR Plot
    sns.boxplot(
        x='Type', 
        y='Metric', 
        data=psnr_df,
        width=0.3,
        showfliers=False,
        linewidth=1.2,
        palette=['lightgray', 'skyblue'],
        ax=axes[0]
    )
    axes[0].set_title('PSNR', pad=10)
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_xlabel('')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # SSIM Plot
    sns.boxplot(
        x='Type', 
        y='Metric', 
        data=ssim_df,
        width=0.3,
        showfliers=False,
        linewidth=1.2,
        palette=['lightgray', 'lightgreen'],
        ax=axes[1]
    )
    axes[1].set_title('SSIM', pad=10)
    axes[1].set_ylabel('SSIM')
    axes[1].set_xlabel('')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    # Overall title
    fig.suptitle(f'Image Quality Metrics Before and After Denoising\n({noise_label}, {param_label})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'metrics_comparison_{filename_suffix}.png'), dpi=400, bbox_inches='tight')
    plt.close()


def calculate_error_map(original, denoised):
    """
    Calculate the absolute error map between original and denoised images
    
    Args:
        original: Original clean image (torch tensor)
        denoised: Denoised image (torch tensor)
        
    Returns:
        Error map as numpy array
    """
    original_np = original.squeeze().cpu().numpy()
    denoised_np = denoised.squeeze().cpu().numpy()
    
    error_map = np.abs(original_np - denoised_np)
    
    return error_map


def test_dncnn(args):
    """
    Testing function for DnCNN model with hardcoded noise parameters for each noise type
    """
    # define noise parameters for each noise type
    noise_params = {
        "gaussian": [10, 20, 25], 
        "rician": [10, 20, 25],    
        "gibbs": [0.6, 0.7, 0.8],  
        "spike": [0.2, 0.3, 0.4]   
    }
    
    # select the appropriate parameters based on noise type
    test_params = noise_params[args.noise_type]
    
    # Create base directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model
    results_name = f"{model_name}_{args.learning}_{args.noise_type}"
    results_base_dir = os.path.join(args.results_dir, results_name)
    os.makedirs(results_base_dir, exist_ok=True)

    # test directory (timestamped)
    test_dir = os.path.join(results_base_dir, f"test_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)

    # subdirectories
    metrics_dir = os.path.join(test_dir, "metrics")
    images_dir = os.path.join(test_dir, "images")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # ;ogging
    logging.basicConfig(filename=os.path.join(test_dir, 'test_results.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DnCNN(channels=1, num_of_layers=args.num_layers, features=args.num_features)
    print("Using DnCNN")
    logging.info("Using DnCNN")

    model = model.to(device)

    #load model weights
    model_path = find_best_model(args.save_dir)
    if model_path is None:
        print(f"No model found in {args.save_dir}")
        logging.error(f"No model found in {args.save_dir}")
        return

    print(f"Loading model from {model_path}")
    logging.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    logging.info(f'Total parameters: {total_params}')
    
    # check learning mode from checkpoint if available
    saved_learning_mode = args.learning
    if 'learning_mode' in checkpoint:
        saved_learning_mode = checkpoint['learning_mode']
        print(f"Using learning mode from checkpoint: {saved_learning_mode}")
        logging.info(f"Using learning mode from checkpoint: {saved_learning_mode}")
    else:
        print(f"Using specified learning mode: {saved_learning_mode}")
        logging.info(f"Using specified learning mode: {saved_learning_mode}")

    # data loader for test set for clean images
    test_loader = create_data_loaders(
        root_dir=args.data_dir,
        splits_file=args.splits_file,
        noise_type=args.noise_type,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        split="test",
        slice_low=args.slice_low,
        slice_high=args.slice_high
    )

    # loop over each noise parameter
    for param_value in test_params:

        if args.noise_type == "gaussian" or args.noise_type == "rician":
            param_desc = f"σ={param_value}"
        elif args.noise_type == "gibbs":
            param_desc = f"α={param_value:.1f}"
        elif args.noise_type == "spike":
            param_desc = f"intensity={param_value:.1f}"
        
        print(f"\n--- Testing with {args.noise_type.capitalize()} noise: {param_desc} ---\n")
        logging.info(f"Testing with {args.noise_type.capitalize()} noise: {param_desc}")

        model.eval()
        results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing {param_desc}")):
                if batch['clean'] is None:
                    continue

                clean = batch['clean'].to(device)
                filenames = batch['filename']

                # apply fixed deterministic noise based on noise type
                noisy = apply_fixed_noise(clean, args.noise_type, param_value, batch_idx, args.seed)

                # inference
                output = model(noisy)

                for i in range(clean.size(0)):
                    c = clean[i:i+1]
                    n = noisy[i:i+1]
                    o = output[i:i+1]

                    # process based on learning type
                    if saved_learning_mode == "noise":
                        # model predicts noise, so subtract from noisy to get denoised
                        d = torch.clamp(n - o, 0, 1)
                        noise_pred = o
                    else:
                        # model directly predicts clean image
                        d = torch.clamp(o, 0, 1)
                        noise_pred = n - d  

                    # compute metrics
                    psnr_noisy, ssim_noisy = calculate_metrics(c, n)
                    psnr_denoised, ssim_denoised = calculate_metrics(c, d)

                    # Format parameter for the filename
                    if args.noise_type == "gaussian" or args.noise_type == "rician":
                        param_suffix = f"sigma{param_value}"
                    elif args.noise_type == "gibbs":
                        param_suffix = f"alpha{param_value:.1f}"
                    elif args.noise_type == "spike":
                        param_suffix = f"intensity{param_value:.1f}"

                    results.append({
                        'filename': filenames[i],
                        'psnr_noisy': psnr_noisy,
                        'psnr_denoised': psnr_denoised,
                        'psnr_gain': psnr_denoised - psnr_noisy,
                        'ssim_noisy': ssim_noisy,
                        'ssim_denoised': ssim_denoised,
                        'ssim_gain': ssim_denoised - ssim_noisy,
                        'param_value': param_value
                    })

        # save metrics for this noise level
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(metrics_dir, f'results_{param_suffix}.csv'), index=False)

        # summary
        summary = df[['psnr_noisy', 'psnr_denoised', 'psnr_gain',
                    'ssim_noisy', 'ssim_denoised', 'ssim_gain']].mean()
        print(f"\n[{param_desc}] Summary:\n{summary}")
        logging.info(f"[{param_desc}] Summary:\n{summary}")

        # Create plots
        create_metrics_plots(df, metrics_dir, args.noise_type, param_value)

    # Save parameters
    with open(os.path.join(test_dir, 'test_parameters.txt'), 'w') as f:
        f.write("DnCNN Testing Parameters\n")
        f.write("========================\n\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write(f"\nHardcoded Noise Parameters:\n")
        f.write(f"  {args.noise_type}: {test_params}\n")
        f.write(f"\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DnCNN for MRI Denoising - Testing')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='/cluster/home/t134723uhn/LowFieldMRI/data_v2/split2', 
                        help='Path to data directory')
    parser.add_argument('--splits_file', type=str, default='/cluster/home/t134723uhn/LowFieldMRI/data_v2/split2/splits_Guys.json', 
                        help='Path to the splits file')
    parser.add_argument('--learning', type=str, default='clean', choices=['clean', 'noise', 'noise2noise'],
                        help="Specify the prediction task: 'clean' (denoise), 'noise' (predict noise), 'noise2noise'")

    # Model parameters
    parser.add_argument('--model', type=str, default='dncnn', 
                        choices=['dncnn', 'dncnn-multi', 'dncnn-skip', 'dncnn-robust'], 
                        help='Model architecture to use')
    parser.add_argument('--num_layers', type=int, default=17, 
                        help='Number of layers in DnCNN')
    parser.add_argument('--num_features', type=int, default=64, 
                        help='Number of feature maps in each layer')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for testing')
    parser.add_argument('--patch_size', type=int, default=64, 
                        help='Size of patches for testing')
    parser.add_argument('--noise_type', type=str, default='gaussian', 
                        choices=['gaussian', 'gibbs', 'rician', 'spike'], 
                        help='Type of noise in test data')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./results', 
                        help='Directory containing trained model')
    parser.add_argument('--results_dir', type=str, default='./test_results', 
                        help='Directory to save test results')
    parser.add_argument('--slice_low', type=int, default=60, 
                        help='Lower slice index threshold')
    parser.add_argument('--slice_high', type=int, default=150, 
                        help='Upper slice index threshold')
    parser.add_argument('--vis_interval', type=int, default=50, 
                        help='Interval for saving visualizations (every n-th sample)')
    
    args = parser.parse_args()
    
    # Run testing
    test_dncnn(args)