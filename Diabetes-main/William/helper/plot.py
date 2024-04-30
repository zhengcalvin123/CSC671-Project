import numpy as np
import torch
import matplotlib.pyplot as plt

# Load CSV file into numpy array
data1 = np.genfromtxt('diabetes_data_convert.csv', delimiter=',', skip_header=1)

# Convert numpy array to Torch Tensor
data_tensor1 = torch.tensor(data1)

# Extract age(first) and labels(last) columns
ages1 = data_tensor1[:, 0]
labels = data_tensor1[:, -1]

# Separate positive and negative classes
positive_ages = ages1[labels == 1].numpy()  # Convert to numpy array for plotting
negative_ages = ages1[labels == 0].numpy()  # Convert to numpy array for plotting

# Define the bins
bins = np.arange(0, 101, 20)

# Plot the histogram
plt.hist([positive_ages, negative_ages], bins=bins, histtype='bar', label=['Positive', 'Negative'], color=['green', 'red'])

plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age')
plt.xticks(bins)
plt.grid(axis='y', alpha=0.75)
plt.legend()
plt.show()
