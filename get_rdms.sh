#!/bin/bash

#WORKSTATION PATHS FOR RDMS
python3 /home/clionaodoherty/cmc_associations/get_rdms.py \
    --image_path /data/imagenet_cmc/to_test \
    --activation_path /data/movie-associations/activations/segmentation/background_only/sup \
    --rdm_path /home/clionaodoherty/cmc_associations/rdms/segmentation/background_only \
    --save_rdm True

