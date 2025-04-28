#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 4
#SBATCH -t 0-4   # time in d-hh:mm:ss
#SBATCH -G 1
#SBATCH -p htc      # partition 
#SBATCH -q public
#SBATCH -A class_eee59820978spring2025
#SBATCH -o logs/slurm/%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/slurm/%j.err # file to save job's STDERR (%j = JobId)

module load mamba/latest
source activate fairml

python scripts/points_train.py \
  --batch_size 2048 \
  --steps 1000 \
  --max_steps 100000 \
  --noise_schedule cosine \
  --n_samples 200000 \
  --r1 2.0 --r2 4.0 --std 0.05

python scripts/eval_metrics.py \
  --n_real 10000 \
  --ckpt checkpoints/ema_0.9999_100000.pt \
  --steps 1000 \
  --n_gen 10000 \
  --width 256
