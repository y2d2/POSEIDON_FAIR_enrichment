import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import Experimental_Setup
import matplotlib.pyplot as plt

# ---------- utilities ----------
def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quat_mul(q, r):
    # Quaternion multiply q * r, both as [w, x, y, z]
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_rotate(q, v):
    # rotate vector(s) v by quaternion q. q = [w,x,y,z]. v shape (...,3)
    q = normalize_quat(q)
    v = np.asarray(v)
    if v.ndim == 1:
        v = v[None, :]
        single = True
    else:
        single = False
    # q * (0, v) * q_conj
    qv = np.concatenate([np.zeros((v.shape[0],1)), v], axis=1)  # (N,4)
    q_conj = q * np.array([1.0, -1.0, -1.0, -1.0])
    # left multiply: for many vectors convert quaternion to matrix for speed
    # but we'll use quaternion multiplication per-vector for clarity:
    out = []
    for vec in v:
        tmp = quat_mul(q, np.concatenate([[0.0], vec]))
        rotated = quat_mul(tmp, q_conj)[1:]
        out.append(rotated)
    out = np.vstack(out)
    return out[0] if single else out

def make_time_weights(timestamps, kind='exp', param=5.0):
    """
    timestamps: 1D array in increasing order corresponding to reference sequence.
    weight function decays with (t - t0)/(tN - t0) so beginning (small t) has high weight.
    """
    t = np.asarray(timestamps, dtype=float)
    if len(t) == 1:
        return np.array([1.0])
    tn = (t - t[0]) / (t[-1] - t[0])
    if kind == 'exp':
        return np.exp(-param * tn)
    elif kind == 'poly':
        return (1.0 + tn) ** (-param)
    elif kind == 'linear':
        w = 1.0 - param * tn
        w[w < 0] = 0.0
        return w
    else:
        raise ValueError("unknown kind")

# ---------- residual function for optimizer ----------
def make_residual_function(A_times, A_pos, B_times, B_pos,
                           weight_kind='exp', weight_param=5.0,
                           dt_bounds=None, allow_extrapolate=False):
    """
    Returns a function f(params) -> residuals (1D array) suitable for least_squares.
    params = [tx, ty, tz, q0, q1, q2, q3, dt]
      - quaternion will be normalized internally.
    A_times, A_pos: reference trajectory times & positions (N,3)
    B_times, B_pos: other trajectory times & positions (M,3)
    weight_kind/param: for time weighting (applied to A_times)
    dt_bounds: (min_dt, max_dt) used for masking out-of-overlap or penalizing extrapolation.
    allow_extrapolate: if False, times outside B_times are masked and residuals removed.
    """
    A_times = np.asarray(A_times)
    A_pos = np.asarray(A_pos)
    B_times = np.asarray(B_times)
    B_pos = np.asarray(B_pos)

    # interpolation for B position (per axis)
    interp_x = interp1d(B_times, B_pos[:,0], kind='linear', bounds_error=False, fill_value=np.nan)
    interp_y = interp1d(B_times, B_pos[:,1], kind='linear', bounds_error=False, fill_value=np.nan)
    interp_z = interp1d(B_times, B_pos[:,2], kind='linear', bounds_error=False, fill_value=np.nan)

    weights_time = make_time_weights(A_times, kind=weight_kind, param=weight_param)
    # sqrt weights applied to residuals (so squared sum is weighted)
    sqrt_w = np.sqrt(weights_time)

    def residuals(params):
        tx, ty, tz = params[0:3]
        q = np.array(params[3:7], dtype=float)
        dt = float(params[7])
        q = normalize_quat(q)

        sample_times = A_times - dt
        bx = interp_x(sample_times)
        by = interp_y(sample_times)
        bz = interp_z(sample_times)
        B_sampled = np.vstack([bx, by, bz]).T  # (N,3)

        # identify valid samples
        valid = ~np.isnan(B_sampled).any(axis=1)

        # rotate and translate B
        B_rot = quat_rotate(q, np.nan_to_num(B_sampled))
        B_trans = B_rot + np.array([tx, ty, tz])

        # residuals for all samples
        res = (A_pos - B_trans)  # (N,3)

        # set residuals for invalid samples to zero or large penalty
        res[~valid, :] = 0.0  # or e.g. 1e3 to penalize missing overlap

        # weight
        wvec = sqrt_w[:, None]
        res_weighted = (wvec * res).ravel()

        # optional dt bound penalty
        penalty = []
        if dt_bounds is not None:
            min_dt, max_dt = dt_bounds
            if dt < min_dt:
                penalty.append(min_dt - dt)
            elif dt > max_dt:
                penalty.append(dt - max_dt)
        if penalty:
            res_weighted = np.concatenate([res_weighted, np.array(penalty) * 1e2])

        return res_weighted

    return residuals

# ---------- helper: coarse dt initial estimate ----------
def coarse_dt_estimate(A_times, A_pos, B_times, B_pos, max_shift_seconds=None, step=0.01, weight_kind='exp', weight_param=5.0):
    """
    Simple coarse search over dt values to find a starting dt.
    Searches dt in [-max_shift_seconds, +max_shift_seconds] with step and returns best dt (min weighted MSE).
    If max_shift_seconds is None it defaults to 20% of duration of combined sequences.
    """
    A_times = np.asarray(A_times)
    if max_shift_seconds is None:
        dur = max(A_times[-1] - A_times[0], B_times[-1] - B_times[0])
        max_shift_seconds = 0.2 * dur
    dts = np.arange(-max_shift_seconds, max_shift_seconds + 1e-12, step)
    best_dt = 0.0
    best_score = np.inf
    weights = make_time_weights(A_times, kind=weight_kind, param=weight_param)
    interp_x = interp1d(B_times, B_pos[:,0], bounds_error=False, fill_value=np.nan)
    interp_y = interp1d(B_times, B_pos[:,1], bounds_error=False, fill_value=np.nan)
    interp_z = interp1d(B_times, B_pos[:,2], bounds_error=False, fill_value=np.nan)
    for dt in dts:
        sample_times = A_times - dt
        bx = interp_x(sample_times); by = interp_y(sample_times); bz = interp_z(sample_times)
        B_sampled = np.vstack([bx,by,bz]).T
        valid = ~np.isnan(B_sampled).any(axis=1)
        if valid.sum() == 0:
            continue
        diff = A_pos[valid] - B_sampled[valid]
        mse = np.sum(weights[valid] * np.sum(diff**2, axis=1)) / (np.sum(weights[valid]) + 1e-12)
        if mse < best_score:
            best_score = mse
            best_dt = dt
    return best_dt, best_score

# ---------- main fitter ----------
def fit_transform_and_dt(A_times, A_pos, B_times, B_pos,
                         initial_params=None,
                         dt_bounds=None,
                         weight_kind='exp',
                         weight_param=5.0,
                         allow_extrapolate=False,
                         robust_loss='soft_l1',
                         max_nfev=2000):
    """
    Fit (tx,ty,tz,q0,q1,q2,q3,dt) using least_squares.
    Returns result (scipy OptimizeResult) and helper info.
    """
    # initial guesses
    if initial_params is None:
        # translation: difference of first points
        t0 = A_pos[0] - B_pos[0]
        q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
        # coarse dt estimate
        dt0, _ = coarse_dt_estimate(A_times, A_pos, B_times, B_pos, weight_kind=weight_kind, weight_param=weight_param)
        dt0 = np.clip(dt0, dt_bounds[0] if dt_bounds else -np.inf, dt_bounds[1] if dt_bounds else np.inf)
        initial_params = np.concatenate([t0, q0, [dt0]])

    residual_fn = make_residual_function(A_times, A_pos, B_times, B_pos,
                                         weight_kind=weight_kind, weight_param=weight_param,
                                         dt_bounds=dt_bounds, allow_extrapolate=allow_extrapolate)

    # bounds for params: no bounds on translation/quat (quat normalized internally), but dt can be bounded
    lower = [-np.inf]*8
    upper = [ np.inf]*8
    if dt_bounds is not None:
        lower[7] = dt_bounds[0]
        upper[7] = dt_bounds[1]

    result = least_squares(residual_fn, initial_params, jac='2-point', bounds=(lower, upper),
                           loss=robust_loss, max_nfev=max_nfev, verbose=2)

    # normalize quaternion in result
    res_params = result.x.copy()
    res_params[3:7] = normalize_quat(res_params[3:7])

    # estimate covariance (approx) if jac available
    cov = None
    if result.jac is not None and result.jac.size > 0 and result.success:
        # approximate covariance of params via (J^T J)^-1 * residual_variance
        try:
            J = result.jac
            JTJ = J.T.dot(J)
            # regularize a tiny bit
            eps = 1e-12 * np.eye(JTJ.shape[0])
            cov = np.linalg.inv(JTJ + eps) * (np.sum(result.fun**2) / max(1, (len(result.fun) - len(res_params))))
        except Exception:
            cov = None

    info = {
        'initial_params': initial_params,
        'final_params': res_params,
        'cost': result.cost,
        'fun': result.fun,
        'jac': result.jac,
        'cov': cov,
        'success': result.success,
        'message': result.message
    }
    return result, info

def load_data():
    call_positions = [[0.300, 3.581, 2.834],
                      [0.028, 4.370, 0.164],
                      [0.201, 1.048, 3.057],
                      [0.144, 0.335, 0.303],
                      [31.705, 3.190, 2.700],
                      [4.825, 1.047, 2.600],
                      [8.005, 4.384, 1.641],
                      [8.073, -0.570, 3.123],
                      [11.968, 1.101, 1.270],
                      [12.123, 2.944, 2.656],
                      [15.805, 4.269, 0.287],
                      [16, 4.5 - 0.89, 3],  # 11 not present in large data set.
                      [15.931, -0.062, -0.020],
                      [15.796, 1.103, 2.700],
                      [7.977, -8.877, 1.950],
                      [7.948, 13.200, 2.100], ]
    anchors_ids = {f"{i}": call_positions[i] for i in range(16)}
    exp_setup = Experimental_Setup.Experiment(True)
    folder_path = "../ESP_code/data/Long Experiment noon"
    folder_path = "../ESP_code/data/Aerolytics_small_extract"
    # folder_path = "../ESP_code/data/small_extract"
    exp_setup.load_data(folder_path)
    exp_setup.set_anchors(anchors_ids)
    load_path = "../ESP_code/data/Aerolytics_small_extract_GT"
    exp_setup.load_gt(load_path)
    exp_setup.smoothingfilter(window_size=0.5)
    exp_setup.load_vio_data("../ESP_code/data/Long Experiment noon VIO/data.csv", 16)
    A_times =exp_setup.tag_gts[exp_setup.tag_gts["id"]==16]["time"].to_numpy()
    print(A_times[1]*1e-3)
    A_times = (A_times[1:] - A_times[1]) * 1e-3  # to seconds
    A_pos = exp_setup.tag_gts[exp_setup.tag_gts["id"]==16][["px","py","pz"]].to_numpy()
    A_pos = A_pos[1:,:]
    # remove all NAN rows:
    while np.any(np.isnan(A_pos[0])):
        A_pos = A_pos[1:,:]
        A_times = A_times[1:]
    B_times = exp_setup.vio_data["timestamp(ns)"].to_numpy()
    print(B_times[0]*1e-9)
    B_times = (B_times - B_times[0]) * 1e-9  # to seconds
    B_pos = exp_setup.vio_data[["T_imu_wrt_vio_x(m)","T_imu_wrt_vio_y(m)","T_imu_wrt_vio_z(m)"]].to_numpy()

    return A_times[:15000], A_pos[:15000], B_times[:2500], B_pos[:2500]

# ---------- Example usage ----------
if __name__ == "__main__":
    # Example synthetic data
    np.random.seed(1)
    # Reference A: times and a 3D curve
    A_times = np.linspace(0, 10, 201)
    A_pos = np.column_stack([np.sin(A_times), np.cos(A_times*0.5), 0.5*np.sin(2*A_times)])  # (201,3)

    # B is same curve rotated, translated and delayed by dt_true
    dt_true = 0.73
    true_t = np.array([0.5, -0.2, 0.8])
    # small rotation about z axis 30 degrees
    angle = np.deg2rad(30)
    qw = np.cos(angle/2)
    qz = np.array([qw, 0.0, 0.0, np.sin(angle/2)])
    # rotate A to make B (inverse transform)
    B_times = A_times + dt_true
    B_pos = quat_rotate(qz, A_pos) - true_t  # note subtract to create translation difference
    # add noise
    B_pos += np.random.normal(scale=0.01, size=B_pos.shape)

    A_times, A_pos, B_times, B_pos = load_data()

    # Fit
    result, info = fit_transform_and_dt(
        A_times, A_pos, B_times, B_pos,
        dt_bounds=(-60.0, 60.0),
        weight_kind='exp',
        weight_param=6.0,
        allow_extrapolate=False,
        robust_loss='soft_l1'
    )

    print("\nFitted params (tx,ty,tz,q0,q1,q2,q3,dt):")
    print(info['final_params'])
    print("Optimization success:", info['success'], info['message'])
    if info['cov'] is not None:
        print("Parameter covariance diagonal:", np.sqrt(np.diag(info['cov'])))

    # Apply fitted transform to B
    tx, ty, tz = info['final_params'][0:3]
    q_fit = info['final_params'][3:7]
    dt_fit = info['final_params'][7]

    B_times_fit = B_times + dt_fit
    B_pos_rot = quat_rotate(q_fit, B_pos)
    B_pos_fit = B_pos_rot + np.array([tx, ty, tz])

    # plot A and B on 3d plot:
    plt.figure().add_subplot(111, projection='3d')
    plt.plot(A_pos[:,0], A_pos[:,1], A_pos[:,2], 'b.-', label='A (ref)')
    plt.plot(B_pos[:,0], B_pos[:,1], B_pos[:,2], 'r.-', label='B (raw)')
    plt.plot(B_pos_fit[:,0], B_pos_fit[:,1], B_pos_fit[:,2], 'g.-', label='B (fitted)')
    plt.show()