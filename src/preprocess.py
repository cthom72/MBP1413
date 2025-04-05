#!/usr/bin/env python
import os
import argparse
import random
import json
import nibabel as nib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_patient_institution(filename):
    """
    Gets patient institution (i.e. "Guys" as some of the institutions have different magnnetic field strengths)
    """

    # parse the string
    basename = os.path.basename(filename)
    name_parts = basename.split('.')[0].split('-')
    if len(name_parts) >= 2:
        patient_id = name_parts[0]
        institution = name_parts[1]
        return patient_id, institution
    else:
        return basename, "Unknown"

def get_available_institutions(data_dir):
    """
    Lists all the possible institutions in the dataset (specifically for IXI)
    """
    institutions = set()
    for filename in os.listdir(data_dir):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            _, institution = get_patient_institution(filename)
            institutions.add(institution)
    return sorted(list(institutions))


def extract_sagittal_slices(nifti_file, output_dir, patient_id, institution):
    """
    Processes specifically the sagittal slices of the image
    """
    # loda image
    img = nib.load(nifti_file)

    # get nity header
    data = img.get_fdata()

    # get affine matrix
    affine = img.affine

    # use the affine matrix to figure out orientation
    abs_affine = np.abs(affine[:3, :3])
    sagittal_axis = np.argmax(abs_affine[0, :])
    coronal_axis = np.argmax(abs_affine[1, :])
    axial_axis = np.argmax(abs_affine[2, :])

    flip_si = affine[2, axial_axis] < 0
    flip_ap = affine[1, coronal_axis] < 0

    # get number of possible sagittal slices
    num_slices = data.shape[sagittal_axis]

    # iterate over all slices depending on sagittal axis
    for i in range(num_slices):
        if sagittal_axis == 0:
            slice_data = data[i, :, :]
            si_axis, ap_axis = 1, 0
        elif sagittal_axis == 1:
            slice_data = data[:, i, :]
            si_axis, ap_axis = 1, 0
        else:
            slice_data = data[:, :, i]
            si_axis, ap_axis = 0, 1

        if flip_si:
            slice_data = np.flip(slice_data, si_axis)
        if flip_ap:
            slice_data = np.flip(slice_data, ap_axis)

        # IMPORTANT : Some nifties are flipped so flip them 90 degrees to get a normal orientnation
        slice_data = np.rot90(slice_data, k=1)

        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        if slice_max > slice_min:
            normalized_slice = (slice_data - slice_min) / (slice_max - slice_min)
        else:
            normalized_slice = slice_data

        slice_filename = f"{patient_id}_{institution}_{i:03d}.png"
        output_path = os.path.join(output_dir, slice_filename)
        plt.imsave(output_path, normalized_slice, cmap='gray')

    return num_slices


def generate_dataset_splits(data_dir, save_dir, institution_filter, num_patients, split_ratio, seed=42):
    """
    Generates the data splits that will be used later by dataset.py - takes arguments specified in main()
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    nifti_files = []
    patient_info = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            filepath = os.path.join(data_dir, filename)
            patient_id, institution = get_patient_institution(filename)
            if institution_filter is None or institution == institution_filter:
                nifti_files.append(filepath)
                patient_info[filepath] = {
                    'patient_id': patient_id,
                    'institution': institution
                }

    unique_patients = list(set(info['patient_id'] for info in patient_info.values()))

    if num_patients > len(unique_patients):
        print(f"Warning: Requested {num_patients} patients but only {len(unique_patients)} are available.")
        num_patients = len(unique_patients)

    selected_patients = random.sample(unique_patients, num_patients)
    selected_files = [f for f in nifti_files if patient_info[f]['patient_id'] in selected_patients]

    print(f"Selected {len(selected_patients)} patients with {len(selected_files)} files")

    train_ratio, val_ratio, test_ratio = split_ratio
    total_ratio = train_ratio + val_ratio + test_ratio

    train_count = int(round(num_patients * (train_ratio / total_ratio)))
    val_count = int(round(num_patients * (val_ratio / total_ratio)))
    test_count = num_patients - train_count - val_count

    random.shuffle(selected_patients)

    train_patients = selected_patients[:train_count]
    val_patients = selected_patients[train_count:train_count + val_count]
    test_patients = selected_patients[train_count + val_count:]

    train_files = []
    val_files = []
    test_files = []
    patient_slices = defaultdict(int)

    print("Extracting sagittal slices...")
    for nifti_file in tqdm(selected_files):
        patient_id = patient_info[nifti_file]['patient_id']
        institution = patient_info[nifti_file]['institution']
        num_slices = extract_sagittal_slices(nifti_file, save_dir, patient_id, institution)
        patient_slices[patient_id] += num_slices
        slice_filenames = [f"{patient_id}_{institution}_{i:03d}.png" for i in range(num_slices)]

        if patient_id in train_patients:
            train_files.extend(slice_filenames)
        elif patient_id in val_patients:
            val_files.extend(slice_filenames)
        elif patient_id in test_patients:
            test_files.extend(slice_filenames)

    institution_key = institution_filter if institution_filter else "All"

    # embed the split_info and save
    split_info = {
        "splits": {
            "train": {
                "patient_ids": train_patients,
                "files": train_files,
                "institution_counts": {institution_key: len(train_patients)}
            },
            "val": {
                "patient_ids": val_patients,
                "files": val_files,
                "institution_counts": {institution_key: len(val_patients)}
            },
            "test": {
                "patient_ids": test_patients,
                "files": test_files,
                "institution_counts": {institution_key: len(test_patients)}
            }
        },
        "metadata": {
            "seed": seed,
            "institution": institution_key,
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "test_patients": len(test_patients),
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "split_ratio": split_ratio
        }
    }

    output_json = os.path.join(save_dir, f"splits_{institution_key}.json")
    with open(output_json, 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\nDataset processing complete!")
    print(f"Train: {len(train_patients)} patients, {len(train_files)} slices")
    print(f"Val: {len(val_patients)} patients, {len(val_files)} slices")
    print(f"Test: {len(test_patients)} patients, {len(test_files)} slices")
    print(f"\nSplit information saved to: {output_json}")

if __name__ == "__main__":

    # arguments to generate data splits
    parser = argparse.ArgumentParser(description='Generate sagittal slices from 3D Nifty files with dataset splits')

    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .nii files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save PNG slices and split information')
    parser.add_argument('--institution', type=str, default=None, help='Optional institution to filter by')
    parser.add_argument('--num_patients', type=int, default=100, help='Number of patients to select')
    parser.add_argument('--split_ratio', type=str, default='70 15 15', help='Train/val/test split ratios, space-separated (e.g., "70 15 15")')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # check to make sure 
    split_parts = [int(x) for x in args.split_ratio.split()]

    generate_dataset_splits(
        args.data_dir,
        args.save_dir,
        args.institution,
        args.num_patients,
        tuple(split_parts),
        args.seed
    )

# python preprocess.py --data_dir /cluster/home/t134723uhn/LowFieldMRI/parkinsons_data_nii --save_dir /cluster/home/t134723uhn/LowFieldMRI/parkinsons_dataset_png --num_patients 20 --split_ratio "0 0 100" 