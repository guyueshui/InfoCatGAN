#!/bin/bash
python launch.py \
  --gan_type='infogan' \
  --experiment_tag='fuck' \
  --num_epoch=5 \
  --num_noise_dim=128 \
  --instance_noise \
  --perform_classification
