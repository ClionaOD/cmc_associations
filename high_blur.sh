#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/blur_sigma10_kernel31_%j.out
#SBATCH --error=/home/clionaodoherty/logs/blur_sigma10_kernel31_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_activations.py \
   --model_path /home/clionaodoherty/cmc_associations/weights \
   --save_path  /data/movie-associations/activations/blurring/sigma10_kernel31 \
   --image_path /data/imagenet_cmc/to_test \
   --transform distort \
   --blur True \
   --sigma 10.0 \
   --kernel_size 31