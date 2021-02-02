#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/segment_bg_sup_%j.out
#SBATCH --error=/home/clionaodoherty/logs/segment_bg_sup_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_activations.py \
   --model_path /home/clionaodoherty/cmc_associations/weights \
   --save_path  /data/movie-associations/activations/segmentation/background_only/sup \
   --image_path /data/imagenet_cmc/to_test \
   --transform distort \
   --segment rm_bg \
   --supervised True
