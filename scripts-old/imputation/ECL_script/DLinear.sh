export CUDA_VISIBLE_DEVICES=0

#!/bin/bash

model_name=${MODEL_NAME}
batch_size=${BATCH_SIZE}
seq_len=${SEQ_LEN}
pred_len=${PRED_LEN}
epochs=${EPOCHS}

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data custom \
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
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --data custom \
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
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data custom \
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
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data custom \
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
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001
