#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/60_obj-only_%j.out
#SBATCH --error=/home/clionaodoherty/logs/60_obj-only_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_activations.py \
   --model_path /data/movie-associations/weights_for_eval/segmented/new \
   --save_path  /data/movie-associations/activations/segmentation/obj_trained/imgnet/rep_3_correct_color \
   --image_path /data/movie-associations/imagenet_cmc_256/to_test 
   