#!/bin/bash
# 

#WORKSTATION PATHS FOR TESTING
python3 /home/clionaodoherty/cmc_associations/get_activations.py \
    --model_path /home/clionaodoherty/cmc_associations/weights \
    --save_path  /home/clionaodoherty/cmc_associations/activations/test \
    --image_path /home/clionaodoherty/cmc_associations/test_imagenet/ \
    --transform distort 

#WORKSPACE PATHS FOR TESTING SUPERVISED
#python3 /home/clionaodoherty/Desktop/cmc_associations/get_activations.py \
#    --model_path /home/clionaodoherty/Desktop/cmc_associations/weights \
#    --save_path  /home/clionaodoherty/Desktop/test_imagenet/activations \
#    --image_path /home/clionaodoherty/test_imagenet/imagenet_categories \
#    --transform distort \
#    --supervised True

#WORKSTATION PATHS
#python3 /home/clionaodoherty/cmc_associations/get_activations.py \
#   --model_path /home/clionaodoherty/cmc_associations/weights \
#   --save_path  /home/clionaodoherty/cmc_associations/activations \
#   --image_path /data/imagenet_cmc \
#   --transform distort \
#   --supervised False

#WORKSTATION PATHS FOR SUPERVISED
#python3 /home/clionaodoherty/cmc_associations/get_activations.py \
#   --model_path /home/clionaodoherty/cmc_associations/weights \
#   --save_path  /home/clionaodoherty/cmc_associations/activations \
#   --image_path /data/imagenet_cmc \
#   --transform distort \
#   --supervised True

