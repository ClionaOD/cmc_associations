#!/bin/bash
# 

#WORKSTATION PATHS FOR TESTING
python3 /home/clionaodoherty/cmc_associations/lch_comparison.py \
    --image_path /home/clionaodoherty/cmc_associations/test_imagenet/ \
    --save_path /home/clionaodoherty/cmc_associations/results/test/test_results.pickle \
    --rdm_path /home/clionaodoherty/cmc_associations/rdms/test \
    --open_lch True \
    --lch_path /home/clionaodoherty/cmc_associations \
    --correlation mantel