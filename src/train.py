import os
import time, datetime
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from dataset import create_data_loaders
from models import DnCNN
import logging
import torch.nn.functional as F

# define the MSE loss function
class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super(SumSquaredErrorLoss, self).__init__()
    
    def forward(self, input, target):
        return F.mse_loss(input, target, reduction='mean') / 2

# writes the parmater files
def save_parameters_to_file(args, filepath):
    """
    Save parameters to a text file
    Args:
        args: Command line arguments
        filepath: Path to save the parameters file
    """
    with open(filepath, 'w') as f:
        # write header
        f.write("DnCNN Training Parameters\n")
        f.write("========================\n\n")
        
        # write all arguments
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        # add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\nTimestamp: {timestamp}\n")


def find_latest_checkpoint(save_dir):
    """
    Find the latest checkpoint file in the save directory
    Args:
        save_dir: Directory containing checkpoint files
    Returns:
        Path to the latest checkpoint file or None if no checkpoint exists
    """
    # ;ook for checkpoint files with pattern checkpoint_epoch_*.pth
    checkpoint_files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    
    if not checkpoint_files:
        # check if best_model.pth or final_model.pth exists
        if os.path.exists(os.path.join(save_dir, 'best_model.pth')):
            return os.path.join(save_dir, 'best_model.pth')
        elif os.path.exists(os.path.join(save_dir, 'final_model.pth')):
            return os.path.join(save_dir, 'final_model.pth')
        else:
            return None
    
    # extract epoch numbers from filenames
    epoch_numbers = []
    for file in checkpoint_files:
        try:
            # extract the epoch number from the filename
            epoch = int(os.path.basename(file).split('_')[-1].split('.')[0])
            epoch_numbers.append((epoch, file))
        except:
            continue
    
    # sort by epoch number and return the path to the latest one
    if epoch_numbers:
        epoch_numbers.sort(reverse=True)
        return epoch_numbers[0][1]
    
    return None


def train_dncnn(args):
    """
    Training function for DnCNN model
    Args:
        args: Command line arguments containing training parameters
    """
    # create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # set up logging
    logging.basicConfig(filename=os.path.join(args.save_dir, 'training.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    
    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # save parameters to a text file
    params_filepath = os.path.join(args.save_dir, 'parameters.txt')
    save_parameters_to_file(args, params_filepath)
    logging.info(f"Parameters saved to {params_filepath}")
    print(f"Parameters saved to {params_filepath}")
    
    # initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # create data loaders
    train_loader = create_data_loaders(
        root_dir=args.data_dir,
        splits_file=args.splits_file,
        noise_type=args.noise_type,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        split="train",
        augment_train=args.augment_data
    )
    
    val_loader = create_data_loaders(
        root_dir=args.data_dir,
        splits_file=args.splits_file,
        noise_type=args.noise_type,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        split="val"
    )
    
    # initialize model based on args
    if args.model == "dncnn":
        model = DnCNN(channels=1, num_of_layers=args.num_layers, features=args.num_features)
        print("Using DnCNN")

    model = model.to(device)
    
    # loss function
    criterion = SumSquaredErrorLoss()
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # training statistics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 30  # stop if no improvement for 30 consecutive epochs
    start_epoch = 0
    
    # check for existing checkpoints
    checkpoint_path = find_latest_checkpoint(args.save_dir)
    if checkpoint_path is not None and args.resume:
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        logging.info(f"Found checkpoint at {checkpoint_path}. Loading...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training statistics if available
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
        if 'val_loss' in checkpoint and checkpoint['val_loss'] < best_val_loss:
            best_val_loss = checkpoint['val_loss']
        
        # determine the starting epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        
        print(f"Resuming training from epoch {start_epoch}")
        logging.info(f"Resuming training from epoch {start_epoch}")
    
    # define the scheduler
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    
    # print a model summary
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total parameters: {total_params}')
    print(f'Total parameters: {total_params}')
    
    # ----training loop---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        # iterate over batches
        for batch_idx, batch in enumerate(train_loader):
            if batch['clean'] is None or batch['noisy'] is None:
                continue
            
            # get clean and noisy
            clean = batch['clean'].to(device)
            noisy = batch['noisy'].to(device)
            
            #zero grad
            optimizer.zero_grad()
            
            if args.learning == 'noise':
                # predict the noise, subtract model output from noisy input
                # Tthe noise is now directly computed as the difference between noisy and clean
                noise = noisy - clean  # This is the actual noise applied by the dataset
                target = noise
                output = model(noisy)
                loss = criterion(output, target)
    
            else:
                raise ValueError("Invalid --learning mode. Choose from 'clean', 'noise', or 'noise2noise'")
            
            # backward propogation
            loss.backward()
            
            # GRAD CLIPPING
            if args.grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            # print progress
            if (batch_idx + 1) % args.print_freq == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.6f}")
        
        # avg training loss for this epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # validation Phase 
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch['clean'] is None or batch['noisy'] is None:
                    continue
                
                clean = batch['clean'].to(device)
                noisy = batch['noisy'].to(device)
                
                output = model(noisy)
                
                if args.learning == 'noise':
                    target = noisy - clean
                    loss = criterion(output, target)
                
                val_loss += loss.item()
        
        # Average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate PSNR for validation set
        val_psnr = calculate_psnr(avg_val_loss)
        
        # End of epoch
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] completed in {epoch_time:.2f}s - "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val PSNR: {val_psnr:.2f}dB")
        logging.info(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, Val PSNR: {val_psnr:.2f}dB, "
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0  # reset counter if improved
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_psnr': val_psnr,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'noise_type': args.noise_type,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            
            print(f"Best model saved with Val Loss: {best_val_loss:.6f}")
            logging.info(f"Best model saved with Val Loss: {best_val_loss:.6f}")
        else:
            early_stopping_counter += 1
            
            # Only log early stopping if enabled
            if args.early_stopping:
                print(f"No improvement in Val Loss. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                logging.info(f"No improvement in Val Loss. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    logging.info("Early stopping triggered.")
                    break
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'noise_type': args.noise_type,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # save the final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'noise_type': args.noise_type,
    }, os.path.join(args.save_dir, 'final_model.pth'))

def calculate_psnr(mse, max_value=1.0):
    """
    Calculate PSNR from MSE
    
    Args:
        mse: Mean squared error
        max_value: Maximum value of the signal (default: 1.0 for normalized images)
        
    Returns:
        PSNR value in dB
    """
    if mse == 0:
        return 100
    return 20 * np.log10(max_value) - 10 * np.log10(mse)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DnCNN for MRI Denoising')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='/cluster/home/t134723uhn/LowFieldMRI/data_v2/split2',
                        help='Path to data directory')
    parser.add_argument('--splits_file', type=str, default='/cluster/home/t134723uhn/LowFieldMRI/data_v2/split2/splits_Guys.json',
                        help='Path to the splits file')
    
    parser.add_argument('--noise_type', type=str, default='gaussian', 
                        choices=['gaussian', 'gibbs', 'rician', 'spike'],
                        help='Type of noise to apply to images')
    parser.add_argument('--noise_level', type=str, default='medium', 
                        choices=['low', 'medium', 'high'],
                        help='Amount of noise to apply (low/medium/high)')
    
    # Model parameters

    parser.add_argument('--num_layers', type=int, default=17,
                        help='Number of layers in DnCNN')
    parser.add_argument('--num_features', type=int, default=64,
                        help='Number of feature maps in each layer')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay parameter')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[5, 10, 15],
                        help='Epochs at which to reduce learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.7,
                        help='Learning rate decay factor')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Frequency of printing training progress')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Frequency of saving checkpoints')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from latest checkpoint if available')
    parser.add_argument('--grad_clipping', action='store_true', default=True,
                        help='Apply gradient clipping during training')
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Enable early stopping')
    
    # Patch and augmentation parameters
    parser.add_argument('--augment_data', action='store_true', default=False,
                        help='Enable patch extraction and data augmentation')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Size of patches extracted from images')
    
    args = parser.parse_args()

    # RUN TRAIN LOOP----
    train_dncnn(args)