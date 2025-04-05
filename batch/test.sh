#!/bin/bash
#SBATCH --job-name=dncnn-mri-test
#SBATCH --output=logs/test/test_%j.out
#SBATCH --error=logs/test/test_%j.err
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --account=#####
#SBATCH --partition=####
#SBATCH -C "gpu32g"

# Load environment
source ~/.bash_profile

conda activate base

# directory
cd /cluster/home/t134723uhn/LowFieldMRI/

# Run training
python -u src/test.py \
    --noise_type gaussian \
    --save_dir /cluster/home/t134723uhn/LowFieldMRI/results/small_dataset_model_v14_dncnn_gaussian \
    --results_dir ./test_results/parkinsons_dataset_gaussian \
    --num_features 48 \
    --vis_interval 20 \
    --data_dir /cluster/home/t134723uhn/LowFieldMRI/parkinsons_dataset_png \
    --splits_file /cluster/home/t134723uhn/LowFieldMRI/parkinsons_dataset_png/splits_All.json