#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 1 # Number of CPU Cores
#SBATCH -p gpus # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 12288 # memory pool for all cores
#SBATCH --nodelist monal05 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log

# Source virtual environment (pip)
source /vol/biomedic3/kc2322/env/activate

# Run python script
python3 main.py -m "unet_v3" -b 6 -n 50 -s True
