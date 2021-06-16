#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/seg_imgnet_bold_sup_%j.out
#SBATCH --error=/home/clionaodoherty/logs/seg_imgnet_bold_sup_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_bold_activations.py \
   --model_path /data/movie-associations/weights_for_eval/segmented \
   --save_path  /data/movie-associations/activations/segmentation/obj_trained/imgnet_bold \
   --image_path /data/movie-associations/imgnet_BOLD5000_1916 \
   --supervised True