-------------
INFORMATION
-------------
MODEL RUNNING
-------------

1. cd /LowFieldMRI/
2. Edit train scripts inside "batch"

Example of params to edit for train.sh (all params have defaults set in src/train.py):

--data_dir: /cluster/home/t134723uhn/LowFieldMRI/data
--splits_file: /cluster/home/t134723uhn/LowFieldMRI/data/splits_100.json
--num_layers: 17
--num_features: 64
--batch_size: 128
--epochs: 40
--lr: 0.001
--weight_decay: 0.0005
--lr_milestones: [5, 10, 15]
--lr_gamma: 0.2
--seed: 42
--num_workers: 4
--save_dir: ./results/small_dataset_model_v5_skip
--print_freq: 100
--save_freq: 10
--noise_level: 0.2
--resume: True
--grad_clipping: True
--augment_data: True
--patch_size: 60
--stride: 20
--model: dncnn-skip

3. To train model, run command in SLURM-compatible sever "sbatch batch/train.sh"

ex. output: Submitted batch job 1882719

4. Training logs can be followed with correspoding SLURM "JOBID" in "logs/train"

ex. log filename: train_1882719

5. Logs show the loss function over batches and at epochs.

ex. training log output:

Extracting patient information...
Filtering images...

train set: 400 patients, 100268 slices
Institution distribution in train set:
  Guys: 222 patients
  HH: 127 patients
  IOP: 51 patients
Extracting patient information...
Filtering images...

val set: 50 patients, 12899 slices
Institution distribution in val set:
  Guys: 27 patients
  HH: 15 patients
  IOP: 8 patients
Total parameters: 667008
Epoch [1/50], Batch [10/784], Loss: 20.101603
Epoch [1/50], Batch [20/784], Loss: 8.419108
Epoch [1/50], Batch [30/784], Loss: 3.068777
Epoch [1/50], Batch [40/784], Loss: 1.342643
Epoch [1/50], Batch [50/784], Loss: 0.718997
Epoch [1/50], Batch [60/784], Loss: 0.465735


6. To test model, run command in SLURM compatible server "sbatch batch/test.sh" (you will need to manually configure where the test script looks for the model). All models are saved in file location specified in the --save_dir param from train.sh.

7. Testing logs can be followed with corresponding SLURM "JOBID" in "logs/test"

8. Testing results are saved at the directory location in /test_results

-------------
DATA DIRECTORIES
-------------

/batch -> contains the shell scripts to run the test and train python files
/logs -> stores logs from the test and train scripts in /batch
/results -> stores the results from training (.pth files, etc) after running train shell scripts in /batch
/src -> contains the code 
  dataset.py -> the dataset class
  models.py -> contains the architecture for the DnCNN
  noise_strength_comparison.py -> used to evaluate comparative strengths between differnet types of noise (to know what similar model difficulties are)
  preprocess.py -> to preprocess .nii files into .png and to save the splits
  test.py -> to test models on the test split
  train.py -> train loop for the denoising
/test_results -> stores the test results after running test shell scripts in /batch
