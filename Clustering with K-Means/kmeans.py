import matplotlib.pyplot as mpl
import numpy as np
from sklearn.datasets import load_iris

def function_visualize(X, centroids, labels, title):
    mpl.figure(figsize=(8, 6))
    mpl.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    mpl.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, c="pink", label="Centroids")
    mpl.title(title)
    mpl.xlabel("Feature 1")
    mpl.ylabel("Feature 2")
    mpl.legend()
    mpl.show()

def kmeans_fnctn(X, K, max_iters=100, tol=1e-4):
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]

    for iteration in range(max_iters):
        space = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(space, axis=1)

        centroids_next = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        if np.linalg.norm(centroids_next - centroids) < tol:
            break

        centroids = centroids_next

        if X.shape[1] == 2:
            function_visualize(X, centroids, labels, f"Iteration {iteration + 1}")


iris = load_iris()
X = iris.data[:, :2] 


centroids = kmeans_fnctn(X, 3)
#print("Centroids:", centroids)
