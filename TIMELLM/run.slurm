#!/bin/bash
#SBATCH --job-name=TS
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --partition=gpu-l40s
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --gpus=1


# Create directories for SLURM logs and outputs
output_path=output

mkdir -p slurm_logs
mkdir -p $output_path

# Define arrays for models, datasets, and prediction lengths
# models=(Autoformer DLinear FEDformer Informer PatchTST TimeMixer TimesNet)
models=(TimeLLM)

# datasets=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic illness)
pred_lens=(720)
illness_pred_lens=(24 36 48 60)

datasets=(ETTm1 ETTm2)
# pred_lens=(720)
# illness_pred_lens=(24)


# Calculate the indices
total_models=${#models[@]}
total_datasets=${#datasets[@]}
total_pred_lens=${#pred_lens[@]}

model_index=$((SLURM_ARRAY_TASK_ID / (total_datasets * total_pred_lens)))
dataset_index=$(((SLURM_ARRAY_TASK_ID % (total_datasets * total_pred_lens)) % total_datasets))
pred_len_index=$(((SLURM_ARRAY_TASK_ID % (total_datasets * total_pred_lens)) / total_datasets))

# Get the model, dataset, and prediction length
model_name=${models[$model_index]}
dataset_name=${datasets[$dataset_index]}
if [ "$dataset_name" == "illness" ]; then
  pred_len=${illness_pred_lens[$pred_len_index]}
else
  pred_len=${pred_lens[$pred_len_index]}
fi

export MODEL_NAME=$model_name

# Print the current MODEL_NAME
echo "Current MODEL_NAME: $MODEL_NAME"

# Create the output directory
mkdir -p $output_path/${model_name}/${dataset_name}

# Call the appropriate script
script_path="scripts/${model_name}/${dataset_name}/${pred_len}.sh"

if [ -f "$script_path" ]; then
  echo "Running $script_path for $model_name on $dataset_name with pred_len $pred_len"
  bash "$script_path" > ${output_path}/${model_name}/${dataset_name}/${model_name}_${dataset_name}_${pred_len}.log 2>&1
else
  echo "Error: Script $script_path not found"
  exit 1
fi