#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Print the values of the arguments
echo "Argument 1 (project): $1"
echo "Argument 2 (use_wandb): $2"
echo "Argument 3 (uid): $3"
echo "Argument 4 (optimizer): $4"
echo "Argument 5 (seed): $5"
echo "Argument 6 (lr): $6"
echo "Argument 7 (lr_decay): $7"
echo "Argument 8 (wd): $8"
echo "Argument 9 (beta1): $9"


python train_imagenet.py --project $1 --use_wandb $2 --uid $3 --optimizer $4 --seed $5 --lr $6 --lr_decay $7 --wd $8 --momentum $9
