#!/bin/bash

model_name=${MODEL_NAME}
batch_size=${BATCH_SIZE}
seq_len=${SEQ_LEN}
pred_len=${PRED_LEN}
epochs=${EPOCHS}

python -u run.py \
  --task_name long_term_forecast \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_300_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --itr 1