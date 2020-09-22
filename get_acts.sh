#!/bin/bash
#
python3 /home/clionaodoherty/cmc_associations/get_acts.sh \
    --model_path /data/movie-associations/saves/temporal/finetune5min/movie-training-distorted/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_128_sec_300_view_temporal/ckpt_epoch_80.pth \
    --save_path /home/clionaodoherty/cmc_associations/activations/5min_distort_testCategs.pickle \
    --image_path /data/imagenet_cmc \
    --transform distort