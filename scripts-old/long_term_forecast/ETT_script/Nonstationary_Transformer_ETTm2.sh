export CUDA_VISIBLE_DEVICES=4

model_name=Nonstationary_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_300_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 256 256 \
  --p_hidden_layers 4

python -u run.py \
  --task_name long_term_forecast \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_300_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 256 256 \
  --p_hidden_layers 4

python -u run.py \
  --task_name long_term_forecast \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_300_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 16 16 16 16 \
  --p_hidden_layers 4

python -u run.py \
  --task_name long_term_forecast \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_300_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 16 16 16 16 \
  --p_hidden_layers 4