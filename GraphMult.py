import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv("matrix_multiplication_results.csv")

# Separate data for each block size
block_sizes = [2, 5, 10, 25, 32]
fig, axes = plt.subplots(1, len(block_sizes), figsize=(20, 5))

for i, block_size in enumerate(block_sizes):
    block_data = data[data["Block Width"] == block_size]
    axes[i].plot(block_data["NumBlocks/BlockWidth"], block_data["GPU Execution Time (ms)"], marker='o')
    axes[i].set_title(f"Block Size {block_size}")
    axes[i].set_xlabel("NumBlocks/BlockWidth")
    axes[i].set_ylabel("GPU Execution Time (ms)")

plt.tight_layout()
plt.show()
