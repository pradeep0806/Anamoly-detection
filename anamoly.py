import numpy as np
import random
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(0)

# Generate data from a normal distribution
data = np.random.normal(0, 1, 1000)

# Introduce multiple anomalies (e.g., 3 anomalies)
num_anomalies = 3
anomaly_indices = random.sample(range(20, 1000), num_anomalies)
anamolyvalue = [20,50,32]
# Set the values at these indices to a large outlier value
for idx, value in zip(anomaly_indices, anamolyvalue):
    data[idx] = value


def detect_anomalies(data, threshold=3):
    """
    Detect anomalies in data based on a given threshold.

    Parameters:
        data (array-like): The input data to analyze.
        threshold (float): The number of standard deviations from the mean to consider an anomaly.

    Returns:
        anomalies (ndarray): Boolean array indicating the positions of anomalies.
        anomaly_indices (ndarray): Indices of the detected anomalies.
        anomaly_values (ndarray): Values of the detected anomalies.
    """
    mean = np.mean(data)
    std = np.std(data)
    anomalies = np.abs(data - mean) > threshold * std
    anomaly_indices = np.where(anomalies)[0]
    anomaly_values = data[anomalies]

    return anomalies, anomaly_indices, anomaly_values


# Detect anomalies
anomalies, detected_anomaly_indices, anomaly_values = detect_anomalies(data)

# Plot the data and highlight the anomalies
plt.figure(figsize=(10, 6))
plt.plot(data, 'b-', label='Data Points')
plt.plot(detected_anomaly_indices, anomaly_values, 'ro', label='Anomalies')
plt.title('Data Points and Anomalies')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()

'''
three types of anomaly exist
point 
contextual 
collective
'''