import torch
# import cupy as cp
# import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(X, Y):
    """
        Cosine distance: d(X, Y) = 1 - (X·Y) / (||X|| ||Y||)
        """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    dot = torch.dot(X_t, Y_t)
    normX = torch.norm(X_t)
    normY = torch.norm(Y_t)
    # Guard against division by zero (if necessary)
    return (1 - dot / (normX * normY)).item()

def distance_l2(X, Y):
    """
        L2 (Euclidean) distance: d(X, Y) = sqrt(sum_i (X_i - Y_i)^2)
        """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    return torch.norm(X_t - Y_t).item()


def distance_dot(X, Y):
    """
       Dot product: d(X, Y) = X·Y
       """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    return torch.dot(X_t, Y_t).item()

def distance_manhattan(X, Y):
    """
        Manhattan (L1) distance: d(X, Y) = sum_i |X_i - Y_i|
        """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    return torch.sum(torch.abs(X_t - Y_t)).item()

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    """
       Given a dataset A (N x D) and a query vector X, compute the L2 distance
       from each vector in A to X and return the indices of the K closest vectors.
       """
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    # Compute the L2 distance between X and every row in A.
    # (No need to compute the square root if only ordering is needed,
    #  but here we compute the full L2 norm for clarity.)
    diff = A_tensor - X_tensor  # shape: (N, D)
    distances = torch.norm(diff, dim=1)  # shape: (N,)

    # Get the indices corresponding to the smallest distances.
    _, indices = torch.topk(distances, k=K, largest=False)
    return indices.cpu().numpy()

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    """
       Perform k-means clustering on A (N x D) with K clusters.
       Returns the final cluster labels (an array of length N).
       """
    data = torch.tensor(A, dtype=torch.float32, device=device)
    # Randomly initialize centroids by picking K data points.
    indices = torch.randperm(N)[:K]
    centroids = data[indices]
    max_iters = 100

    for it in range(max_iters):
        # Compute squared L2 distances between each point and each centroid.
        # data: (N, D) --> (N, 1, D), centroids: (K, D) --> (1, K, D)
        diff = data.unsqueeze(1) - centroids.unsqueeze(0)  # shape: (N, K, D)
        distances = torch.sum(diff ** 2, dim=2)  # shape: (N, K)
        labels = torch.argmin(distances, dim=1)  # shape: (N,)

        # Update centroids: mean of all points assigned to each cluster.
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            if (labels == k).sum() > 0:
                new_centroids[k] = data[labels == k].mean(dim=0)
            else:
                # If a cluster loses all points, keep its previous centroid.
                new_centroids[k] = centroids[k]

        # Check for convergence (if centroids do not change significantly).
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break

        centroids = new_centroids

    # Return the cluster label for each data point.
    return labels.cpu().numpy()

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    """
       A simple ANN search:
         1. Cluster A into a small number of clusters (e.g. 10).
         2. Find the cluster whose centroid is closest to X.
         3. Perform an exact search within that cluster.

       If the best cluster is empty, falls back to exact KNN.
       """
    num_clusters = min(10, N)
    # Cluster A using k-means.
    labels = our_kmeans(N, D, A, num_clusters)

    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    # Compute centroids for each cluster.
    centroids = []
    for k in range(num_clusters):
        mask = (labels_tensor == k)
        if mask.sum() > 0:
            centroids.append(A_tensor[mask].mean(dim=0))
        else:
            centroids.append(torch.zeros(D, device=device))
    centroids = torch.stack(centroids, dim=0)

    # Find the closest cluster to X.
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    diff = centroids - X_tensor
    centroid_distances = torch.norm(diff, dim=1)
    best_cluster = torch.argmin(centroid_distances).item()

    # Get all points belonging to the best cluster.
    mask = (labels_tensor == best_cluster)
    if mask.sum() == 0:
        # Fallback: if no point is found, use exact KNN search.
        return our_knn(N, D, A, X, K)

    cluster_indices = torch.nonzero(mask, as_tuple=False).squeeze()
    # Ensure cluster_indices is 1-D (if only one point is found).
    if cluster_indices.dim() == 0:
        cluster_indices = cluster_indices.unsqueeze(0)

    points_in_cluster = A_tensor[mask]
    distances = torch.norm(points_in_cluster - X_tensor, dim=1)
    effective_K = min(K, points_in_cluster.shape[0])

    # Get the top-K nearest indices within the best cluster.
    knn_indices_in_cluster = torch.topk(distances, k=effective_K, largest=False).indices
    result = cluster_indices[knn_indices_in_cluster].cpu().numpy()
    return result

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# # Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)

def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_kmeans()
