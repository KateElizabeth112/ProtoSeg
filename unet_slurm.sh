#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 4 # Number of CPU Cores
#SBATCH -p gpushigh # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 20G # memory pool for all cores
#SBATCH --nodelist monal04 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log

# Source virtual environment (pip)
source /vol/biomedic3/kc2322/env/bin/activate

# Run python script
python3 main.py -m "unet_v4_0" -b 8 -n 50 -s True -f 0

python3 main.py -m "unet_v4_1" -b 8 -n 50 -s True -f 1

python3 main.py -m "unet_v4_2" -b 8 -n 50 -s True -f 2

python3 main.py -m "unet_v4_3" -b 8 -n 50 -s True -f 3

python3 main.py -m "unet_v4_4" -b 8 -n 50 -s True -f 4