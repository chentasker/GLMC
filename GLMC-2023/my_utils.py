#%% Generate rotation matrix between 2 vectors
def rotation_matrix_from_vectors(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Returns a rotation matrix R (d x d) that rotates vector v to vector u.
    Both v and u must be unit vectors (normalized).
    """
    d = v.shape[0]
    assert v.shape == u.shape, "v and u must be the same shape"
    
    # Dot product and angle
    cos_theta = torch.dot(v, u)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical stability
    theta = torch.acos(cos_theta)
    
    # If vectors are almost identical: no rotation needed
    if torch.isclose(cos_theta, torch.tensor(1.0)):
        return torch.eye(d)
    
    # If vectors are opposite
    if torch.isclose(cos_theta, torch.tensor(-1.0)):
        # Need to find a vector orthogonal to v to build a rotation plane
        # Find any vector orthogonal to v
        orth = None
        for i in range(d):
            basis = torch.zeros(d)
            basis[i] = 1.0
            proj = basis - torch.dot(basis, v) * v
            if torch.norm(proj) > 1e-6:
                orth = proj / torch.norm(proj)
                break
        if orth is None:
            # This would only happen if v is zero vector, which shouldn't happen
            raise ValueError("Cannot find orthogonal vector for opposite vectors.")
        
        # Rotation by 180° in the plane spanned by v and orth
        R = torch.eye(d)
        # Reflection in that plane (same as rotation by pi)
        R -= 2 * orth[:, None] @ orth[None, :]
        return R
    
    # Build orthonormal basis for 2D rotation plane
    a = v
    b = u - cos_theta * v
    b = b / torch.norm(b)
    
    # Construct rotation matrix in plane spanned by a and b
    # R = I + (cosθ - 1)(a a^T + b b^T) + sinθ (b a^T - a b^T)
    
    A = torch.outer(a, a)
    B = torch.outer(b, b)
    abT = torch.outer(a, b)
    baT = torch.outer(b, a)
    
    R = torch.eye(d) + (cos_theta - 1) * (A + B) + torch.sin(theta) * (baT - abT)
    return R

#%% Scatter plot Kzs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def scatter_pca_pairs(X, label, rows=2, cols=2):
    pca = PCA()
    X_pca = pca.fit_transform(X)
    num_plots = rows * cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    unique_labels = sorted(set(label))
    cmap = plt.get_cmap('tab10')  # Or use 'viridis', 'Set1', 'Dark2', etc.
    label_to_color = {val: cmap(i) for i, val in enumerate(unique_labels)}
    color_values = [label_to_color[val] for val in label]
    
    for i in range(num_plots):
        if i + 1 >= X_pca.shape[1]:
            break
        ax = axes[i]
        sc = ax.scatter(X_pca[:, 2*i], X_pca[:,2*i+1], s=10, alpha=0.8, c=color_values, cmap=cmap)
        ax.set_xlabel(f'PC{i+1}')
        ax.set_ylabel(f'PC{i+2}')
        ax.set_title(f'PC{2*i} vs PC{2*i+1}')
    #plt.legend(*sc.legend_elements(), title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=label_to_color[l], label=str(l))
               for l in unique_labels]
    ax.legend(handles=handles, title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

"""
scatter_pca_pairs(np.concatenate(list(kz_tensor_for_coarse.values()), axis=0),
                  np.concatenate([coarse_labels[[int(t) for t in list(targets_for_coarse[k])]] for k in targets_for_coarse.keys()])
                  )
"""
coarse_label = 14
scatter_pca_pairs(kz_tensor_for_coarse[coarse_label],targets_for_coarse[coarse_label])
print(acc_per_coarse[coarse_label])

#%% Create coarse labels by kz energy distance

def _energy_distance_empirical(X1, X2):
    E12 = np.mean([np.linalg.norm(x1 - x2) for x1 in X1 for x2 in X2])
    E11 = np.mean([np.linalg.norm(x1 - x2) for x1 in X1 for x2 in X1])
    E22 = np.mean([np.linalg.norm(x1 - x2) for x1 in X2 for x2 in X2])
    return 2 * E12 - E11 - E22

first_pc_for_class = {}
for c, X in kz_tensor_for_class.items():
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    first_pc = eigenvectors[:, -1]  # shape: (d,)
    first_eigenvalue = eigenvalues[-1]
    first_pc_for_class[c] = first_pc

n_classes = len(coarse_labels)
distance_table = np.zeros((n_classes, n_classes))
for c1 in range(n_classes):
    for c2 in range(c1, n_classes):
        print(f'{c1}, {c2}')
        
        distance_table[c1,c2] = distance_table[c2,c1] = 1-np.dot(first_pc_for_class[c1], first_pc_for_class[c2])

import matplotlib.pyplot as plt
import numpy as np

plt.imshow(distance_table, cmap='viridis', aspect='auto')
plt.colorbar()  # Optional: adds color scale
plt.show()


semantic_table = np.ones((n_classes, n_classes))
for f in fine_labels:
    for i in f:
        for j in f:
            semantic_table[i, j] = 0

plt.imshow(semantic_table, cmap='viridis', aspect='auto')
plt.colorbar()  # Optional: adds color scale
plt.show()

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# D is your NxN distance matrix (must be symmetric with zeros on diagonal)
# Step 1: Convert to condensed form (upper triangle as a flat array)
condensed_D = squareform(distance_table)

# Step 2: Perform hierarchical clustering (e.g., 'average', 'single', 'complete')
Z = linkage(condensed_D, method='single')

# Step 3: Extract clusters (e.g., 3 clusters)
clusters = fcluster(Z, t=30, criterion='maxclust')


from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Assume D is your (N x N) distance matrix
N = distance_table.shape[0]
k = 20  # number of clusters
cluster_size = N // k

# Step 1: Embed into Euclidean space (optional if already in vector form)
from sklearn.manifold import MDS
embedding = MDS(dissimilarity='precomputed', random_state=0).fit_transform(distance_table)

# Step 2: Run k-means to get centroids
kmeans = KMeans(n_clusters=k, random_state=0).fit(embedding)
centroids = kmeans.cluster_centers_

# Step 3: Assign points to clusters with size constraint
from sklearn.metrics.pairwise import euclidean_distances
cost_matrix = euclidean_distances(embedding, centroids)

# Step 4: Enforce fixed cluster size using assignment
assignments = np.full(N, -1)
available = list(range(N))
for cluster_id in range(k):
    sub_cost = cost_matrix[available][:, cluster_id]
    idx_sorted = np.argsort(sub_cost)[:cluster_size]
    chosen = [available[i] for i in idx_sorted]
    for i in chosen:
        assignments[i] = cluster_id
    available = [i for i in available if i not in chosen]