import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

# load data
file_path = 'file1.txt'
data = pd.read_csv(file_path, delimiter='\t')
X = data.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

sorted_indices = np.argsort(pca.explained_variance_ratio_)[::-1]
sorted_variance = pca.explained_variance_ratio_[sorted_indices]
sorted_components = pca.components_[sorted_indices]
cumulative_variance_ratio_sorted = np.cumsum(sorted_variance)

# Find top 90%
index_90_percent = np.argmax(cumulative_variance_ratio_sorted >= 0.9) + 1
top_components = sorted_components[:index_90_percent]

# print(f"Selected Components: {index_90_percent}")
# print(f"Cumulative Explained Variance Ratio: {cumulative_variance_ratio_sorted[index_90_percent - 1]:.2f}")

# Optionally, you can transform the data using the selected components
X_pca_selected = np.dot(X_scaled, top_components.T)

# Perform One-Hot encoding
encoder = OneHotEncoder(sparse=False)
X_onehot = encoder.fit_transform(X_pca_selected)

# Print the shape of the One-Hot encoded data
# print("Shape of One-Hot encoded data:", X_onehot.shape)
