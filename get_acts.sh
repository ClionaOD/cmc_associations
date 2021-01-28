#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/gauss_10_%j.out
#SBATCH --error=/home/clionaodoherty/logs/gauss_10_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_activations.py \
   --model_path /home/clionaodoherty/cmc_associations/weights \
   --save_path  /home/clionaodoherty/cmc_associations/activations/main_redo \
   --image_path /data/imagenet_cmc/to_test \
   --transform distort 