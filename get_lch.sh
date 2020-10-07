#!/bin/bash
# 

#WORKSTATION PATHS FOR TESTING
python3 /home/clionaodoherty/cmc_associations/lch_comparison.py \
    --image_path /data/imagenet_cmc/to_test \
    --save_path /home/clionaodoherty/cmc_associations/results/256_class_results.pickle \
    --rdm_path /home/clionaodoherty/cmc_associations/rdms \
    --open_lch True \
    --lch_path /home/clionaodoherty/cmc_associations \
    --correlation spearman