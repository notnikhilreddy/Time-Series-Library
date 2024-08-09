#!/bin/bash

model_name=${MODEL_NAME}
batch_size=${BATCH_SIZE}
seq_len=${SEQ_LEN}
pred_len=${PRED_LEN}
epochs=${EPOCHS}

train_epochs=${EPOCHS}

learning_rate=0.01
llama_layers=16

master_port=29514
num_process=1
batch_size=${BATCH_SIZE}
d_model=32
d_ff=128

export model_name=${MODEL_NAME}
batch_size=${BATCH_SIZE}
seq_len=${SEQ_LEN}
pred_len=${PRED_LEN}
epochs=${EPOCHS}
export DATASET='ETTm2'
export PRED_LEN=192

comment='TimeLLM-ECL'

mkdir -p output
mkdir -p output/${MODEL_NAME}
mkdir -p output/${MODEL_NAME}/${DATASET}

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_name'_'$DATASET'_300_'$PRED_LEN \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size $batch_size \
  --train_epochs ${EPOCHS} \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment