#!/bin/bash
#SBATCH --job-name=TS
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --partition=gpu-l40s
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --gpus=1

# Create directories for SLURM logs and outputs
output_path=output

mkdir -p slurm_logs
mkdir -p $output_path

# Define arrays for datasets and prediction lengths
# datasets=(electricity ETTh1 ETTh2 ETTm1 ETTm2 traffic weather illness)
datasets=(electricity traffic)
data_paths=(
    "dataset/electricity/electricity.csv"
    # "dataset/ETT-small/ETTh1.csv"
    # "dataset/ETT-small/ETTh2.csv"
    # "dataset/ETT-small/ETTm1.csv"
    # "dataset/ETT-small/ETTm2.csv"
    "dataset/traffic/traffic.csv"
    # "dataset/weather/weather.csv"
    # "dataset/illness/national_illness.csv"
)
# pred_lens=(96 192 336 720)
pred_lens=(720)
illness_pred_lens=(24 36 48 60)

# Calculate the indices
total_datasets=${#datasets[@]}
total_pred_lens=${#pred_lens[@]}

dataset_index=$((SLURM_ARRAY_TASK_ID / total_pred_lens))
pred_len_index=$((SLURM_ARRAY_TASK_ID % total_pred_lens))

# Get the dataset, prediction length, and data path
dataset_name=${datasets[$dataset_index]}
data_path=${data_paths[$dataset_index]}

# Set prediction length based on dataset
if [ "$dataset_name" == "illness" ]; then
  pred_len=${illness_pred_lens[$pred_len_index]}
else
  pred_len=${pred_lens[$pred_len_index]}
fi

# Print the current dataset and prediction length
echo "Current dataset: $dataset_name"
echo "Prediction length: $pred_len"

# Create the output directory
mkdir -p $output_path/Reservoir/${dataset_name}

# Set feature_dim based on dataset
case $dataset_name in
  "electricity") feature_dim=321 ;;
  "ETTh1"|"ETTh2") feature_dim=7 ;;
  "ETTm1"|"ETTm2") feature_dim=7 ;;
  "traffic") feature_dim=862 ;;
  "weather") feature_dim=21 ;;
  "illness") feature_dim=7 ;;
  *) echo "Unknown dataset: $dataset_name"; exit 1 ;;
esac

# Construct the command
cmd="PYTHONUNBUFFERED=1 python -u reservoir.py \
  --epochs 10 \
  --batch_size 4 \
  --data_path $data_path \
  --data $dataset_name \
  --seq_len 300 \
  --feature_dim $feature_dim \
  --pred_len $pred_len \
  --label_len 48 \
  --decoder_dropout 0.00 \
  --inner 2 \
  --num_layers 2 \
  --hidden_size $feature_dim \
  --hidden_dropout_prob 0.0 \
  --reservoir_size 70 \
  --spectral_radius 0.5 \
  --leaky 0.3 \
  --sparsity 0.1 \
  --num_seq_len_heads 12"

# Run the command
echo "Running: $cmd"
eval $cmd > ${output_path}/Reservoir/${dataset_name}/Reservoir_${dataset_name}_${pred_len}.log 2>&1