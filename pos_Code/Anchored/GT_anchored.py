import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def generate_devices(n_devices=16, space_size=10.0, seed=None):
    """
    Generate positions with the first device at the origin.
    """
    if seed is not None:
        np.random.seed(seed)
    positions = np.random.uniform(0.0, space_size, (n_devices, 3))
    positions[0] = [0.0, 0.0, 0.0]  # anchor first device at origin
    return positions

def compute_pairwise_distances(positions):
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)

def compute_noisy_distances(positions, noise_std=0.5, seed=None):
    """
    Compute pairwise distances between devices with Gaussian noise.

    Parameters:
        positions (ndarray): Array of shape (n_devices, 3)
        noise_std (float): Standard deviation of the noise added to each distance
        seed (int or None): Random seed for reproducibility

    Returns:
        distances (ndarray): Noisy pairwise distance matrix (n_devices x n_devices)
    """
    if seed is not None:
        np.random.seed(seed)
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    noise = np.random.normal(0, noise_std, distances.shape)
    noisy_distances = distances + noise
    np.fill_diagonal(noisy_distances, 0.0)  # keep self-distance = 0
    return noisy_distances

def classical_mds(distances, n_components=3):
    n = distances.shape[0]
    D2 = distances ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    L = np.diag(np.sqrt(np.clip(eigvals[:n_components], a_min=0, a_max=None)))
    V = eigvecs[:, :n_components]
    return V @ L

def align_with_anchor(true_positions, recon_positions, anchor_idx=0):
    """
    Align reconstructed positions to true positions using only rotation/scale
    but keep the anchor_idx fixed at the origin.
    """
    # Translate so anchor is at origin in both sets
    t_true = true_positions[anchor_idx]
    t_recon = recon_positions[anchor_idx]
    T = true_positions - t_true
    R = recon_positions - t_recon

    # Solve for best scale+rotation using Umeyama but without translation:
    n, m = T.shape
    sigma2_r = (R ** 2).sum() / n
    cov = (T.T @ R) / n
    U, D, Vt = np.linalg.svd(cov)
    V = Vt.T
    S_mat = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S_mat[-1, -1] = -1.0
    Rmat = U @ S_mat @ V.T
    scale = np.trace(np.diag(D) @ S_mat) / sigma2_r

    recon_aligned = scale * (R @ Rmat.T)  # anchor is at (0,0,0)
    return recon_aligned, scale, Rmat

# Example usage:
true_pos = generate_devices(n_devices=16, space_size=10.0, seed=123)
D = compute_noisy_distances(true_pos)
recon_pos = classical_mds(D, n_components=3)
aligned_recon, scale, Rmat = align_with_anchor(true_pos, recon_pos, anchor_idx=0)

print("True positions:\n", true_pos)
print("\nReconstructed aligned (first device at origin):\n", aligned_recon)
print("\nFirst device (should be at origin):", aligned_recon[0])

# (Optional) 3D plot: true vs aligned reconstructed
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], marker='o')
ax.scatter(aligned_recon[:, 0], aligned_recon[:, 1], aligned_recon[:, 2], marker='^')
for i in range(true_pos.shape[0]):
    ax.plot([true_pos[i, 0], aligned_recon[i, 0]],
            [true_pos[i, 1], aligned_recon[i, 1]],
            [true_pos[i, 2], aligned_recon[i, 2]],
            linewidth=0.5)
ax.set_title("True positions (o) and reconstructed aligned (^) — lines show point-wise residuals")
plt.show()