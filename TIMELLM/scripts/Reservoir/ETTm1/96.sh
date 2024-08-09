#!/bin/bash

python -u reservoir.py \
  --epochs ${EPOCHS} \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --data_path dataset/ETT-small/ETTm1.csv \
  --seq_len $seq_len \
  --feature_dim 7 \
  --pred_len 96 \
  --decoder_dropout 0.00 \
  --inner 2 \
  --num_layers 2 \
  --hidden_size 7 \
  --hidden_dropout_prob 0.0 \
  --reservoir_size 70 \
  --spectral_radius 0.5 \
  --leaky 0.3 \
  --sparsity 0.1 \
  --num_seq_len_heads 12 \