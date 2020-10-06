#!/bin/bash

#WORKSTATION PATHS FOR RDMS 
python3 /home/clionaodoherty/cmc_associations/get_rdms.py \
    --image_path /data/imagenet_cmc/to_test \
    --activation_path /home/clionaodoherty/cmc_associations/activations \
    --rdm_path /home/clionaodoherty/cmc_associations/rdms \
    --save_rdm True