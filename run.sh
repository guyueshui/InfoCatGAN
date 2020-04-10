#!/bin/bash
python launch.py \
  --dataset='CIFAR10' \
  --batch_size=20 \
  --gan_type='catgan' \
  --experiment_tag='fuck' \
  --num_epoch=200 \
  --num_noise_dim=100 \
  --instance_noise \
  --perform_classification
