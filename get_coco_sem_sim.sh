#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/coco_sem_sim_%j.out
#SBATCH --error=/home/clionaodoherty/logs/coco_sem_sim_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2

python3 /home/clionaodoherty/cmc_associations/get_coco_sem_sim.py