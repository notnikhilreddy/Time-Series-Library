#!/bin/bash

model_name=${MODEL_NAME}
batch_size=${BATCH_SIZE}
seq_len=${SEQ_LEN}
pred_len=${PRED_LEN}
epochs=${EPOCHS}

seq_len=312
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=${BATCH_SIZE}


python -u run.py \
  --task_name long_term_forecast \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window