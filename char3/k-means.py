import numpy as np
import matplotlib.pyplot as plt


class KMeans:
	def __init__(self, n_clusters, max_iter=300):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
	
	def fit(self, X):
		self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
		
		for _ in range(self.max_iter):
			# Assign each sample to the nearest centroid
			labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
			
			# Update centroids based on the mean of samples assigned to each centroid
			new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
			
			# Check for convergence
			if np.allclose(self.centroids, new_centroids):
				break
			
			self.centroids = new_centroids
		
		self.labels_ = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
		return self
