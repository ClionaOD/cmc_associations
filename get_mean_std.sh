#!/bin/bash

#SBATCH --output=/home/clionaodoherty/stats_eval_%j.out
#SBATCH --error=/home/clionaodoherty/stats_eval_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

python /home/clionaodoherty/cmc_associations/get_mean_std.py