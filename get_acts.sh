#!/bin/bash
# 

#WORKSTATION PATHS FOR TESTING
python3 /home/clionaodoherty/cmc_associations/get_activations.py \
    --model_path /home/clionaodoherty/cmc_associations/weights \
    --save_path  /home/clionaodoherty/cmc_associations/activations/test \
    --image_path /home/clionaodoherty/cmc_associations/test_imagenet/ \
    --transform distort \
    --supervised True