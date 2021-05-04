#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/main_coco_sup_rep-2_repeat_%j.out
#SBATCH --error=/home/clionaodoherty/logs/main_coco_sup_rep-2_repeat_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_coco_activations.py \
   --model_path /data/movie-associations/weights_for_eval/main \
   --save_path  /data/movie-associations/activations/bold_coco/rep_2 \
   --image_path /data/movie-associations/MSCOCO_BOLD5000_2000 \
   --supervised True