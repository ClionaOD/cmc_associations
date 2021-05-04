#!/bin/bash

#SBATCH --output=/home/clionaodoherty/logs/mantel_bold_main_%j.out
#SBATCH --error=/home/clionaodoherty/logs/mantel_bold_main_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2

R CMD BATCH mantel_test_bold.R