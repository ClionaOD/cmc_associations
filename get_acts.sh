#!/bin/bash
# 

#WORKSTATION PATHS
python3 /home/clionaodoherty/cmc_associations/get_activations.py \
   --model_path /home/clionaodoherty/cmc_associations/weights \
   --save_path  /home/clionaodoherty/cmc_associations/activations \
   --image_path /data/imagenet_cmc/to_test \
   --transform distort 