import re
import csv
import os
import glob

def extract_details(log_content, data, pred_len):
    details = {
        'Model': 'Reservoir',
        'Data': data,
        'Pred Len': pred_len,
        'Seq Len': '',
        'Batch Size': '',
        'Epochs': '',
        'Total Duration': '',
        'Avg Epoch Duration': 0,
        'Test MSE Loss': '',
        'Test MAE Loss': '',
        'Max GPU Utilization': 0,
        'Parameter Count': ''
    }

    # Extract other details
    seq_len_match = re.search(r'seq_len: (\d+)', log_content)
    details['Seq Len'] = seq_len_match.group(1) if seq_len_match else ''
    
    batch_size_match = re.search(r'batch_size: (\d+)', log_content)
    details['Batch Size'] = batch_size_match.group(1) if batch_size_match else ''
    
    epochs_match = re.search(r'epochs: (\d+)', log_content)
    details['Epochs'] = epochs_match.group(1) if epochs_match else ''
    
    duration_match = re.search(r'Duration:\s+(\d+\.\d+)', log_content)
    details['Total Duration'] = duration_match.group(1) if duration_match else ''

    # Extract epoch durations and calculate average
    epoch_durations = re.findall(r'Epoch Duration: (\d+\.\d+)', log_content)
    if epoch_durations:
        details['Avg Epoch Duration'] = sum(float(d) for d in epoch_durations) / len(epoch_durations)

    # Extract last epoch's MSE and MAE
    mse_losses = re.findall(r'Val MSE Loss: (\d+\.\d+)', log_content)
    mae_losses = re.findall(r'Val MAE Loss: (\d+\.\d+)', log_content)
    details['Test MSE Loss'] = mse_losses[-1] if mse_losses else ''
    details['Test MAE Loss'] = mae_losses[-1] if mae_losses else ''

    # Extract max GPU utilization
    gpu_utilizations = re.findall(r'Max GPU Utilization: (\d+\.\d+)', log_content)
    if gpu_utilizations:
        details['Max GPU Utilization'] = max(float(u) for u in gpu_utilizations)

    # Extract parameter count
    param_count_match = re.search(r'Trainable parameters: (\d+)', log_content)
    details['Parameter Count'] = param_count_match.group(1) if param_count_match else ''

    return details

def save_to_csv(all_details, filename='output/output_reservoir.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Data', 'Pred Len', 'Seq Len', 'Batch Size', 'Epochs', 
                      'Total Duration', 'Avg Epoch Duration', 'Test MSE Loss', 
                      'Test MAE Loss', 'Max GPU Utilization', 'Parameter Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for details in all_details:
            writer.writerow(details)

# Process all log files
base_dir = 'output/Reservoir'
all_details = []

for data_dir in os.listdir(base_dir):
    data_path = os.path.join(base_dir, data_dir)
    if os.path.isdir(data_path):
        for log_file in glob.glob(os.path.join(data_path, '*.log')):
            pred_len = os.path.basename(log_file).split('.')[0]
            with open(log_file, 'r') as file:
                log_content = file.read()
            details = extract_details(log_content, data_dir, pred_len)
            all_details.append(details)

# Save all details to CSV
save_to_csv(all_details)

print("Details extracted and saved to model_details.csv")