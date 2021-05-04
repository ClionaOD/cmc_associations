#!/bin/bash
#SBATCH --output=/home/clionaodoherty/logs/rdms_bold_main_%j.out
#SBATCH --error=/home/clionaodoherty/logs/rdms_bold_main_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2

#WORKSTATION PATHS FOR RDMS
python3 /home/clionaodoherty/cmc_associations/get_rdms.py 
 
