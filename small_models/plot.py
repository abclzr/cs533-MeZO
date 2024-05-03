import matplotlib.pyplot as plt

# Data
tasks = ['SST-2', 'SST-5', 'MNLI', 'QNLI', 'SNLI', 'QQP', 'TREC']
finetuning_scores = [0.8956422018348624, 0.443, 0.566, 0.653, 0.622, 0.658, 0.752]
MeZO_scores = [0.8841743119266054, 0.448, 0.558, 0.618, 0.582, 0.67, 0.636]

x = range(len(tasks))

# Plotting
plt.figure(figsize=(10, 6))
bar_width = 0.35
plt.bar(x, finetuning_scores, width=bar_width, label='Fine-tuning')
plt.bar([i + bar_width for i in x], MeZO_scores, width=bar_width, label='MeZO')

# plt.xlabel('Tasks', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.title('Accuracy of Fine-tuning and MeZO across different tasks', fontsize=22)
plt.xticks([i + bar_width / 2 for i in x], tasks, fontsize=22)
plt.legend(fontsize=22)
plt.yticks(fontsize=22)

plt.tight_layout()
plt.savefig('acc.pdf')



# Data
tasks = ['MNLI', 'TREC']
fine_tune_memory = [7030, 6518]  # in MiB
MeZO_memory = [3624, 3624]  # in MiB

x = range(len(tasks))

# Plotting
plt.figure(figsize=(8, 6))
bar_width = 0.35
plt.bar(x, fine_tune_memory, width=bar_width, label='Fine-tuning')
plt.bar([i + bar_width for i in x], MeZO_memory, width=bar_width, label='MeZO')

# plt.xlabel('Tasks')
plt.ylabel('Memory Consumption (MiB)', fontsize=22)
# plt.title('Comparison of Memory Consumption: Finetuning vs MeZO', fontsize=22)
plt.xticks([i + bar_width / 2 for i in x], tasks, fontsize=22)
plt.legend(fontsize=22)
plt.yticks(fontsize=22)

plt.tight_layout()
plt.savefig('memory.pdf')

import pdb
import re
from datetime import datetime

def extract_info(line):
    # Define regex pattern to match the timestamp, loss, and global step
    pattern = r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - .'

    # Search for matches using regex
    match = re.search(pattern, line)

    if match:
        # Extract timestamp, loss, and global step from the match
        timestamp_str, = match.groups()

        # Parse timestamp string to datetime object
        timestamp = datetime.strptime(timestamp_str, '%m/%d/%Y %H:%M:%S')
        loss_index = line.find("'loss':")
        first_space_index = line.find(" ", loss_index)
        second_space_index = line.find(" ", first_space_index + 1)
        loss_value = line[first_space_index:second_space_index].strip()[:-1]

        # Convert loss and global step to float and integer respectively
        loss = float(loss_value)

        return timestamp, loss
    else:
        pdb.set_trace()
        return None, None

def process_file(filename):
    timestamp_list = []
    loss_list = []
    global_step_list = []
    duration_list = []
    with open(filename, 'r') as f:
        for line in f:
            if 'loss' in line and 'learning_rate' in line:
                timestamp, loss = extract_info(line)
                timestamp_list.append(timestamp)
                loss_list.append(loss)
    start_time = timestamp_list[0]
    for timestamp in timestamp_list:
        duration = (timestamp - start_time).total_seconds()
        duration_list.append(duration)
    return duration_list, loss_list

durations1, losses1 = process_file('/common/home/zl606/MeZO/medium_models/slurm-37462.out')
durations2, losses2 = process_file('/common/home/zl606/MeZO/medium_models/finetune_SST-2.out')
length = len(durations2) * 2
plt.figure(figsize=(8, 6))
# Plotting the curve
plt.plot(durations1[:length], losses1[:length], label='MeZO')
plt.plot(durations2, losses2, label='Fine-tuning')

# Adding labels and title
plt.xlabel('Traing time (seconds)', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.legend(fontsize=22)
# plt.title('Loss vs Duration')
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)

# Display the plot
# plt.grid(True)
plt.tight_layout()
plt.savefig('curve.pdf')
