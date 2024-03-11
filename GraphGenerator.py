import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read transfer times from CSV file
df = pd.read_csv('transfer_times.csv')

# Extract data
matrix_sizes = df['Matrix Size'].unique()
avg_host_to_device_times = df.groupby('Matrix Size')['Host to Device (ms)'].mean()
std_host_to_device_times = df.groupby('Matrix Size')['Host to Device (ms)'].std()
avg_device_to_host_times = df.groupby('Matrix Size')['Device to Host (ms)'].mean()
std_device_to_host_times = df.groupby('Matrix Size')['Device to Host (ms)'].std()

# Plot Host to Device transfer times with error bars
plt.errorbar(matrix_sizes, avg_host_to_device_times, yerr=std_host_to_device_times, fmt='-o', label='Host to Device')
plt.title('Host to Device Data Transfer Time vs. Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('Transfer Time (ms)')
plt.grid(True)
plt.legend()
plt.show()

# Plot Device to Host transfer times with error bars
plt.errorbar(matrix_sizes, avg_device_to_host_times, yerr=std_device_to_host_times, fmt='-o', label='Device to Host')
plt.title('Device to Host Data Transfer Time vs. Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('Transfer Time (ms)')
plt.grid(True)
plt.legend()
plt.show()
