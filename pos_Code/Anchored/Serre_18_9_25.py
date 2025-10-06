import numpy as np
from scipy.optimize import least_squares

# -------------------------
# Helpers
# -------------------------
def pairwise_distances_from_positions(positions):
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _vector_to_positions(x, n_devices, free_idx):
    """
    Convert optimization vector x to full (n_devices,3) positions array.
    free_idx is list/array of device indices that are being optimized.
    """
    positions = np.zeros((n_devices, 3), dtype=float)
    # x has length len(free_idx)*3
    positions[free_idx, :] = x.reshape((-1, 3))
    return positions


def _positions_to_vector(positions, free_idx):
    return positions[free_idx, :].ravel()

# -------------------------
# Main fitting function
# -------------------------
def refine_positions_from_distances(
    initial_positions,
    measured_mean,
    measured_std,
    fixed_mask=None,
    exclude_mask=None,
    relative_sigma_floor=1e-3,
    loss='soft_l1',
    verbose=2,
    xtol=1e-8,
    ftol=1e-8,
    max_nfev=2000
):
    """
    Refine 3D positions using pairwise measured distances (means) and per-pair std.

    Parameters
    ----------
    initial_positions : ndarray (n,3)
        Coarse initial positions. For devices you want to optimize, provide their starting guess.
        For fixed devices (anchors) you must either set fixed_mask or ensure their values are in initial_positions.
    measured_mean : ndarray (n,n)
        Measured mean distances between devices. Use np.nan for missing pairs.
    measured_std : ndarray (n,n)
        Measured standard deviation (uncertainty) for each pair. Use np.nan for missing pairs.
    fixed_mask : ndarray (n,), bool or None
        True for devices whose positions are fixed (anchors). If None, no devices are fixed.
    exclude_mask : ndarray (n,n), bool or None
        True for pairs to explicitly exclude (e.g. known NLOS). Excluded pairs are treated as missing.
    relative_sigma_floor : float
        Minimum relative fractional sigma to avoid division by zero. sigma_ij = max(measured_std_ij, relative_sigma_floor)
    loss : str
        Loss function passed to least_squares. 'soft_l1' or 'huber' recommended.
    verbose, xtol, ftol, max_nfev : solver options.

    Returns
    -------
    refined_positions : ndarray (n,3)
        Refined positions (fixed ones unchanged).
    result : OptimizeResult
        The output from scipy.optimize.least_squares
    """

    n = initial_positions.shape[0]
    assert measured_mean.shape == (n, n)
    assert measured_std.shape == (n, n)

    # masks
    if fixed_mask is None:
        fixed_mask = np.zeros(n, dtype=bool)
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool)

    if exclude_mask is None:
        exclude_mask = np.zeros((n, n), dtype=bool)
    else:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)

    # prepare masks for used pairs (upper triangle only)
    iu = np.triu_indices(n, k=1)
    valid_mask = ~np.isnan(measured_mean[iu]) & ~np.isnan(measured_std[iu]) & (~exclude_mask[iu])
    # Build arrays of indices and measured values for faster residual calculation
    idx_i = iu[0][valid_mask]
    idx_j = iu[1][valid_mask]
    meas_d = measured_mean[iu][valid_mask]
    meas_s = measured_std[iu][valid_mask]

    # enforce minimum sigma (avoids division by zero). We use absolute floor based on relative_sigma_floor * mean_distance,
    # so very small distances don't create unrealistically small sigma.
    sigma_floor = np.maximum(relative_sigma_floor * np.maximum(meas_d, 1.0), relative_sigma_floor)
    meas_s_clipped = np.maximum(meas_s, sigma_floor)

    # which device indices are free (to be optimized)
    free_idx = np.where(~fixed_mask)[0]
    fixed_idx = np.where(fixed_mask)[0]

    # initial guess vector
    x0 = _positions_to_vector(initial_positions, free_idx)

    # create boolean arrays mapping each pair to whether both endpoints are free, one fixed, etc.
    both_free = np.isin(idx_i, free_idx) & np.isin(idx_j, free_idx)
    i_free_j_fixed = np.isin(idx_i, free_idx) & np.isin(idx_j, fixed_idx)
    i_fixed_j_free = np.isin(idx_i, fixed_idx) & np.isin(idx_j, free_idx)
    both_fixed = np.isin(idx_i, fixed_idx) & np.isin(idx_j, fixed_idx)  # we can ignore these; they give no info on free unknowns

    # Pre-store fixed positions for quick access
    fixed_positions = np.zeros((n, 3))
    if fixed_idx.size > 0:
        fixed_positions[fixed_idx, :] = initial_positions[fixed_idx, :]

    # residual function: returns residuals vector for least_squares
    def residuals_fn(x):
        positions = _vector_to_positions(x, n, free_idx)
        # fill fixed positions
        if fixed_idx.size > 0:
            positions[fixed_idx, :] = fixed_positions[fixed_idx, :]

        # compute distances only for required pairs
        # option: compute whole distance matrix and index; simpler and fine for n ~ 16
        D = pairwise_distances_from_positions(positions)
        preds = D[idx_i, idx_j]
        res = (preds - meas_d) / meas_s_clipped
        return res

    # run solver with robust loss
    result = least_squares(
        residuals_fn,
        x0,
        loss=loss,
        xtol=xtol,
        ftol=ftol,
        max_nfev=max_nfev,
        verbose=verbose
    )

    # assemble final positions
    refined_positions = _vector_to_positions(result.x, n, free_idx)
    if fixed_idx.size > 0:
        refined_positions[fixed_idx, :] = fixed_positions[fixed_idx, :]

    return refined_positions, result

# -------------------------
# Simple automatic NLOS detector (optional)
# -------------------------
def detect_high_variance_pairs(measured_std, threshold_factor=3.0):
    """
    Mark candidate NLOS/corrupted pairs as True where per-pair std is
    significantly larger than the median std.
    Returns boolean mask (n,n) symmetric with diagonal False.
    """
    n = measured_std.shape[0]
    iu = np.triu_indices(n, k=1)
    st_upper = measured_std[iu]
    median = np.nanmedian(st_upper)
    # guard median==0
    median = max(median, 1e-12)
    mask_upper = st_upper > threshold_factor * median
    mask = np.zeros((n, n), dtype=bool)
    mask[iu] = mask_upper
    mask = mask + mask.T
    np.fill_diagonal(mask, False)
    return mask



if __name__=="__main__":
    # Rough initial positions (somewhat realistic)
    rough_positions = [ [0, 4.5 - 0.86, 3],
                        [0, 4.5 - 0.14, 0.45],
                        [0, 0.94, 3],
                        [0, 0.1 , 0.43],
                        [32, 4.5 -1.35, 3],
                        [8-3.3, 0.99, 3],
                        [8, 4.5 - 0.025, 1.65],
                        [8, -0.87, 3],
                        [8+4.14, 1,97, 3],
                        [8+4.14, 4.5 - 1.3,3],
                        [16, 4.5-0.025, 0.36],
                        [16, 4.5-0.89, 3],
                        [16, 0, 0.28],
                        [16, 1.02, 3],
                        [8, -9, 1.96],
                        [8, 4.5+9, 1.80],
                        ]
    rough_positions = np.array(rough_positions)
    # assume positions_true (n,3) is ground truth (for testing)
    # measured_samples is (n_samples, n, n) from your simulator; you can compute mean/std from it
    mean_D = np.nanmean(measured_samples, axis=0)
    std_D = np.nanstd(measured_samples, axis=0, ddof=1)

    # coarse initial guess: maybe the anchors known roughly, others random or from classical MDS
    initial_guess = coarse_positions.copy()

    # fixed anchors mask: e.g., first 4 are anchors and known exactly
    fixed_mask = np.zeros(n, dtype=bool)
    fixed_mask[:4] = True  # keep first 4 fixed (anchors)

    # try automatic NLOS detection based on variance
    exclude_mask = detect_high_variance_pairs(std_D, threshold_factor=4.0)

    # refine
    refined_positions, res = refine_positions_from_distances(
        initial_positions=initial_guess,
        measured_mean=mean_D,
        measured_std=std_D,
        fixed_mask=fixed_mask,
        exclude_mask=exclude_mask,
        relative_sigma_floor=1e-3,
        loss='soft_l1',
        verbose=1
    )

    print("Optimization success:", res.success, "cost:", res.cost)