#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/main_imgnet_bold_%j.out
#SBATCH --error=/home/clionaodoherty/logs/main_imgnet_bold_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_bold_activations.py \
   --model_path /data/movie-associations/weights_for_eval/main \
   --save_path  /data/movie-associations/activations/bold_imgnet \
   --image_path /data/movie-associations/imgnet_BOLD5000_1916 