#!/bin/bash
#SBATCH --job-name=dncnn-mri-train
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --account=#####
#SBATCH --partition=#####
#SBATCH -C "gpu32g"
# Load environment
source ~/.bash_profile
conda activate base

# directory
cd /cluster/home/t134723uhn/LowFieldMRI/

# run training
python -u src/train.py \
    --num_layers 17 \
    --splits_file /cluster/home/t134723uhn/LowFieldMRI/data_v2/split2/splits_Guys.json \
    --lr 5e-4 \
    --batch_size 64 \
    --epochs 50 \
    --data_dir /cluster/home/t134723uhn/LowFieldMRI/data_v2/split2 \
    --save_dir ./results/small_dataset_model_v14_dncnn_gibbs \
    --num_features 48 \
    --print_freq 50 \
    --lr_milestones 2 5 10 15 20 25 30 35 40 \
    --lr_gamma 0.5 \
    --weight_decay 1e-4 \
    --augment_data \
    --patch_size 90 \
    --noise_type gibbs \
