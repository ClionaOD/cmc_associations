#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/seg_coco_sup_rep-3_repeat_%j.out
#SBATCH --error=/home/clionaodoherty/logs/seg_coco_sup_rep-3_repeat_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_coco_activations.py \
   --model_path /data/movie-associations/weights_for_eval/segmented \
   --save_path  /data/movie-associations/activations/segmentation/obj_trained/coco/rep_3_all \
   --image_path /data/movie-associations/MSCOCO_BOLD5000_2000 \
   --supervised True