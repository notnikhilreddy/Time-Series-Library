import os
import re
import csv
from datetime import datetime

def extract_info(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    def safe_extract(pattern, default='null'):
        match = re.search(pattern, content)
        return match.group(1) if match else default

    # Extract required information
    model_name = safe_extract(r'Model:\s+(\w+)')
    
    # Determine data based on file path
    if "electricity" in file_path:
        data = "electricity"
    elif "illness" in file_path or "ili" in file_path:
        data = "illness"
    elif "weather" in file_path:
        data = "weather"
    elif "traffic" in file_path:
        data = "traffic"
    else:
        data = safe_extract(r'Data:\s+(\w+)')
    
    seq_len = safe_extract(r'Seq Len:\s+(\d+)')
    pred_len = safe_extract(r'Pred Len:\s+(\d+)')
    batch_size = safe_extract(r'Batch Size:\s+(\d+)')
    epochs = safe_extract(r'Train Epochs:\s+(\d+)')
    
    # Extract duration
    start_time_str = safe_extract(r'START TIME =\s+(\d{2}:\d{2}:\d{2})')
    end_time_str = safe_extract(r'END TIME =\s+(\d{2}:\d{2}:\d{2})')
    
    total_duration = 'null'
    avg_epoch_duration = 'null'
    if start_time_str != 'null' and end_time_str != 'null':
        start_time = datetime.strptime(start_time_str, '%H:%M:%S')
        end_time = datetime.strptime(end_time_str, '%H:%M:%S')
        total_duration = (end_time - start_time).total_seconds()
        
        # Extract epoch count (in case early stopping occurred)
        epoch_count = len(re.findall(r'Epoch: \d+', content))
        avg_epoch_duration = total_duration / epoch_count if epoch_count > 0 else 0
        
        total_duration = f"{total_duration:.2f}"
        avg_epoch_duration = f"{avg_epoch_duration:.2f}"
    
    # Extract test losses from the last epoch
    test_mse_losses = re.findall(r'Test MSE Loss: ([\d.]+)', content)
    test_mae_losses = re.findall(r'Test MAE Loss: ([\d.]+)', content)
    test_mse_loss = test_mse_losses[-1] if test_mse_losses else 'null'
    test_mae_loss = test_mae_losses[-1] if test_mae_losses else 'null'
    
    # Extract max GPU utilization
    gpu_utilizations = re.findall(r'Max GPU Utilization: (\d+) MB', content)
    max_gpu_utilization = max(map(int, gpu_utilizations)) if gpu_utilizations else 'null'

    return {
        'Model': model_name,
        'Data': data,
        'Seq Len': seq_len,
        'Pred Len': pred_len,
        'Batch Size': batch_size,
        'Epochs': epochs,
        'Total Duration': total_duration,
        'Avg Epoch Duration': avg_epoch_duration,
        'Test MSE Loss': test_mse_loss,
        'Test MAE Loss': test_mae_loss,
        'Max GPU Utilization': max_gpu_utilization
    }

def main():
    output_dir = 'output'
    csv_file = 'output/output.csv'
    
    results = []

    for model_dir in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model_dir)
        if os.path.isdir(model_path):
            for data_dir in os.listdir(model_path):
                data_path = os.path.join(model_path, data_dir)
                if os.path.isdir(data_path):
                    for log_file in os.listdir(data_path):
                        if log_file.endswith('.log'):
                            file_path = os.path.join(data_path, log_file)
                            results.append(extract_info(file_path))

    # Write results to CSV
    if results:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results have been written to {csv_file}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()

    import time
    now = time.time()
    now = datetime.fromtimestamp(now).strftime('%H:%M:%S')
    print("Current time:", now, flush=True)