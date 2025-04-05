# define imports

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import re
from typing import Tuple
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandRicianNoised,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)

# a custom collage function to include clean and noisy images along with the file name
def custom_collate_fn(batch):
    """
    Custom collate function to handle different image sizes and ensure proper batching
    """
    # Filter out failed samples (where either clean or noisy is None)
    batch = [sample for sample in batch if sample['clean'] is not None and sample['noisy'] is not None]
    
    if len(batch) == 0:
        # If all samples failed, return an empty dict
        return {'clean': None, 'noisy': None, 'filename': []}
    
    # Extract clean and noisy images
    clean_images = [sample['clean'] for sample in batch]
    noisy_images = [sample['noisy'] for sample in batch]
    filenames = [sample['filename'] for sample in batch]
    
    # Stack images into batches
    clean_batch = torch.stack(clean_images)
    noisy_batch = torch.stack(noisy_images)
    
    # define noise strengths
    noise_strengths = [sample['noise_strength'] for sample in batch]
    noise_strengths_tensor = torch.tensor(noise_strengths)

    return {
        'clean': clean_batch,
        'noisy': noisy_batch,
        'filename': filenames,
        'noise_strength': noise_strengths_tensor
    }


class MRIDataset(Dataset):

    # initialize the function
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        splits_file: str = "splits.json",
        noise_type: str = "gaussian",
        noise_params: dict = None,
        augment_data: bool = False,
        patch_size: int = 128,
        patch_overlap: float = 0.5,
        resize_dim: Tuple[int, int] = (180, 180),
        slice_low: int = 20,
        slice_high: int = 130
    ):
        """
        MRI Dataset for DnCNN denoising
        
        Args:
            root_dir: Directory containing images
            split: Data split ('train', 'val', 'test')
            splits_file: JSON file with split information
            noise_type: Type of noise to apply ('gaussian', 'gibbs', 'rician')
            noise_params: Parameters for noise generation
            augment_data: Whether to use data augmentation and patch extraction
            patch_size: Size of patches to extract if augment_data is True
            patch_overlap: Fraction of overlap between patches
            resize_dim: Target dimensions for resizing images
        """
        self.root_dir = root_dir
        self.split = split
        self.noise_type = noise_type
        self.augment_data = augment_data
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - patch_overlap))
        self.resize_dim = resize_dim
        
        # Set default noise parameters if not provided
        if noise_params is None:

            # if gaussian (std/sigma)
            if noise_type == "gaussian":
                self.noise_params = {"mean": 0.0, "std_min": 10, "std_max": 30}

            # if gibbs (alpha)
            elif noise_type == "gibbs":
                self.noise_params = {"alpha_min": 0.1, "alpha_max": 0.5}
            
            # if rician (std/sigma)
            elif noise_type == "rician":
                self.noise_params = {"std_min": 10, "std_max": 30}

            # else error
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
        
        # use predfined noise params if they are there
        else:
            self.noise_params = noise_params
            
        # Load the data split
        with open(splits_file, 'r') as f:
            self.splits_data = json.load(f)

        # Get all files from split
        all_files = self.splits_data["splits"][split]["files"]

        # Filter filenames ending in a number between 20 and 130 (inclusive)
        filtered_files = []

        # iterate all over all files
        for f in all_files:

            # extract slice number
            match = re.search(r"(\d+)(?:\.\w+)?$", f)

            # if a match is found
            if match:

                # use this slice (filter out the otheres to be excluded)
                num = int(match.group(1))
                if slice_low <= num <= slice_high:
                    filtered_files.append(f)

        # use these as the base images
        self.images = filtered_files

        # report numnber of 2D slices being considered in this split        
        print(f"\n{split} set: {len(self.images)} slices")
        
        # Setup transforms for clean images
        self.clean_transform = self._setup_clean_transforms()
                
        # Setup augmentation transforms if enabled
        self.aug_transform = self._setup_augmentation() if augment_data else None
        
        # Precompute all patches if augmentation is enabled
        self.patches = None
        if self.augment_data:

            # generate patches
            self.patches = self._generate_all_patches()
            print(f"Generated {len(self.patches)} patches from {split} set.")
    
    # base transform for clean images
    def _setup_clean_transforms(self):
        """Setup transforms for clean images"""
        return Compose([
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ])
   
    # random augmentations to apply to patches to create data diversity
    def _setup_augmentation(self):
        """Setup augmentation transforms"""
        return Compose([

            # apply a random flip 
            RandFlipd(keys=["image"], spatial_axis=[0, 1], prob=0.5),

            # apply a random rotation
            RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),

            # apply a random contrast enhancement (intensity adjustment)
            RandAdjustContrastd(
                keys=["image"],
                prob=0.75,
                gamma=(0.7, 1.3)
            )
        ])

    # function to generate noise
    def _gen_noise(self, image, idx):
        """
        Generates noise based on Gaussian, Rician, and Gibbs requirements. 

        args: 
        """
        
        # Apply appropriate noise based on noise type
        if self.noise_type == "gaussian":

            # define the std min and amx
            std_min = self.noise_params.get("std_min", 10)
            std_max = self.noise_params.get("std_max", 30)

            # normalize to 0 to 1
            noise_std = np.random.uniform(std_min, std_max) / 255
            noise_mean = self.noise_params.get("mean", 0.0)

            # define the MONAI transform
            noise_transform = RandGaussianNoised(
                keys=["image"],
                prob=1.0,
                mean=noise_mean,
                std=noise_std
            )

            data = {"image": image}
            noisy_data = noise_transform(data)
            noisy_image = torch.clamp(noisy_data["image"], 0, 1)

            # return
            return {
                'clean': image,
                'noisy': noisy_image,
                'filename': f"patch_{idx}",
                'noise_strength': noise_std
            }
    

        elif self.noise_type == "rician":
            std_min = self.noise_params.get("std_min", 10)
            std_max = self.noise_params.get("std_max", 30)
            noise_std = np.random.uniform(std_min, std_max) / 255

            noise_transform = RandRicianNoised(
                keys=["image"],
                prob=1.0,
                std=np.random.uniform(self.noise_params.get("std_min", 8), 
                                self.noise_params.get("std_max", 20)) / 255
            )

            data = {"image": image}
            noisy_data = noise_transform(data)
            noisy_image = torch.clamp(noisy_data["image"], 0, 1)

            return {
                'clean': image,
                'noisy': noisy_image,
                'filename': f"patch_{idx}",
                'noise_strength': noise_std
            }
        
        elif self.noise_type == "gibbs":
            alpha_min = self.noise_params.get("alpha_min", 0.05)
            alpha_max = self.noise_params.get("alpha_max", 0.5)
            alpha = np.random.uniform(alpha_min, alpha_max)

            img_np = image.squeeze().cpu().numpy()
            H, W = img_np.shape
            kspace = np.fft.fft2(img_np)
            kspace_shifted = np.fft.fftshift(kspace)

            max_radius = min(H, W) / 2.0
            radius = (1.0 - alpha) * max_radius

            yy, xx = np.ogrid[:H, :W]
            cy, cx = H // 2, W // 2
            dist_sq = (yy - cy)**2 + (xx - cx)**2
            dist = np.sqrt(dist_sq)

            # Soft sigmoid mask with noise
            falloff = 1.0 / (1.0 + np.exp((dist - radius) / (alpha * 10)))
            noise_mask = np.random.normal(loc=falloff, scale=0.05)
            noise_mask = np.clip(noise_mask, 0.0, 1.0)

            kspace_shifted_masked = kspace_shifted * noise_mask
            kspace_unshifted = np.fft.ifftshift(kspace_shifted_masked)
            img_gibbs = np.abs(np.fft.ifft2(kspace_unshifted))

            img_gibbs = np.clip(img_gibbs, 0, 1)
            noisy_image = torch.tensor(img_gibbs, dtype=image.dtype, device=image.device).unsqueeze(0)
            noise_strength = alpha
            
            return {
                'clean': image,
                'noisy': noisy_image,
                'filename': f"patch_{idx}",
                'noise_strength': alpha
            }

    # patchify the images
    def _generate_all_patches(self):
        """Extract patches from all images with augmentation"""
        patches = []

        # iterate over all the paths
        for img_path in self.images:

            # join the path with the root dir
            full_img_path = os.path.join(self.root_dir, img_path)
            try:
                # Load and preprocess using PIL directly
                with Image.open(full_img_path) as img:
                    img = img.convert('L')
                    img = img.resize(self.resize_dim, Image.BILINEAR)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    
                    # Make sure array has channel dimension
                    img_array = img_array[np.newaxis, ...] 
                    
                    # Extract patches
                    h, w = img_array.shape[1:] 

                    # iterate over height
                    for i in range(0, h - self.patch_size + 1, self.stride):

                        # iterate over width
                        for j in range(0, w - self.patch_size + 1, self.stride):

                            # Extract patch 
                            patch = img_array[:, i:i+self.patch_size, j:j+self.patch_size]
                            
                            # Create dictionary for MONAI transforms
                            patch_data = {"image": torch.from_numpy(patch)}
                            
                            # Apply augmentation transforms if enabled
                            if self.aug_transform:
                                try:
                                    patch_data = self.aug_transform(patch_data)
                                except Exception as aug_error:
                                    print(f"Augmentation error on patch from {img_path}: {aug_error}")
                                    continue
                            
                            # Store the patched tensor
                            patches.append(patch_data["image"])

            except Exception as e:
                print(f"Patch extraction error at {img_path}: {e}")
        
        return patches
    
    # get the length
    def __len__(self):

        # check if patches should be returned
        if self.augment_data and self.patches is not None:
            return len(self.patches)

        # else images
        return len(self.images)
    
    def __getitem__(self, idx):
        try:

            ## PATCHES
            if self.augment_data and (self.patches is not None) and len(self.patches) > 0:
                # If using precomputed patches
                image_tensor = self.patches[idx]
                
                item = self._gen_noise(image_tensor, idx)

                return item

            ## NON PATCHES
            else:

                # get image
                img_name = os.path.join(self.root_dir, self.images[idx])

                # open image
                with Image.open(img_name) as img:

                    # normalize
                    img = img.convert('L')
                    img = img.resize(self.resize_dim, Image.BILINEAR)
                    img_array = np.array(img).astype(np.float32) / 255.0

                # Add channel dimension and convert to tensor
                img_array = img_array[np.newaxis, ...] 
                clean_image = torch.from_numpy(img_array).float()

                # Apply clean transforms
                data = {"image": clean_image}
                data = self.clean_transform(data)
                clean_image = data["image"]

                item = self._gen_noise(clean_image, idx)

                return item


        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return {
                'clean': None,
                'noisy': None,
                'filename': f'Error: {e}',
                'noise_strength': None
            }


# a function to automatically create the necessary dataset (sort of like a handler function)
def create_data_loaders(
    root_dir="/cluster/home/t134723uhn/LowFieldMRI/data_v2/split2",
    splits_file="/cluster/home/t134723uhn/LowFieldMRI/data_v2/split2/splits_Guys.json",
    batch_size=128,
    num_workers=4,
    noise_type="gaussian",
    patch_size=128,
    split="train",
    noise_params=None,
    slice_low=20,
    slice_high=130
):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        root_dir: Directory containing images
        splits_file: JSON file with split information
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        noise_type: Type of noise to apply ('gaussian', 'gibbs', 'rician', 'spike')
        noise_params: Parameters for noise generation
        augment_train: Whether to use data augmentation for training
        patch_size: Size of patches to extract if augment_train is True
        split: "train" "val" or "test"
    
    Returns:
        dict: Dictionary with train, val, and test data loaders
    """

    # Initialize default noise_params based on noise_type if not provided
    if noise_params is None:

        if noise_type == "gaussian":

            noise_params = {"mean": 0.0, "std_min": 10, "std_max": 30}

        elif noise_type == "gibbs":

            noise_params = {"alpha_min": 0.1, "alpha_max": 1.0}

        elif noise_type == "rician":

            noise_params = {"std_min": 10, "std_max": 30}

    # if the split is train
    if (split=="train"):

        # Create datasets
        train_dataset = MRIDataset(
            root_dir=root_dir,
            split='train',
            splits_file=splits_file,
            noise_type=noise_type,
            noise_params=noise_params,
            augment_data=True,
            patch_size=patch_size,
            slice_low=slice_low,
            slice_high=slice_high
        )

        # return dataloader
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    
    # if it is val
    elif(split=="val"):
        val_dataset = MRIDataset(
            root_dir=root_dir,
            split='val',
            splits_file=splits_file,
            noise_type=noise_type,
            noise_params=noise_params,
            augment_data=False,
            slice_low=slice_low,
            slice_high=slice_high
        )

        # return dataloader
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    
    # if it is test
    elif(split=="test"):
        test_dataset = MRIDataset(
            root_dir=root_dir,
            split='test',
            splits_file=splits_file,
            noise_type=noise_type,
            noise_params=noise_params,
            augment_data=False,
            slice_low=slice_low,
            slice_high=slice_high
        )

        # return the dataloader
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )