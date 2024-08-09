#!/bin/bash

# Assign command-line arguments to variables
data_path="\$1"
pred_len=\$2

python -u reservoir.py \
  --epochs ${EPOCHS} \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --data_path $data_path \
  --seq_len $seq_len \
  --feature_dim 7 \
  --pred_len $pred_len \
  --decoder_dropout 0.00 \
  --inner 2 \
  --num_layers 2 \
  --hidden_size 7 \
  --hidden_dropout_prob 0.0 \
  --reservoir_size 70 \
  --spectral_radius 0.5 \
  --leaky 0.3 \
  --sparsity 0.1 \
  --num_seq_len_heads 12