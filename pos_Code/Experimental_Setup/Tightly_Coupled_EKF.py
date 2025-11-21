#!/usr/bin/env python3
"""
tight_coupled_ekf.py

Tightly-coupled EKF that includes orientation (quaternion) in the state and fuses:
 - IMU (gyro, accel) continuously for propagation
 - Range measurements to a fixed anchor for corrections

State vector (16):
  x = [q_w, q_x, q_y, q_z,
       px, py, pz,
       vx, vy, vz,
       bgx, bgy, bgz,
       bax, bay, baz]

Notes:
 - Gyros expected in rad/s, acc in m/s^2, time in seconds.
 - The script uses finite-difference Jacobians for propagation linearization.
 - Initialize from stationary accel (to get initial pitch/roll). If you have a magnetometer
   and want yaw init, incorporate it.
"""
import numpy as np
import pandas as pd
from math import sin, cos
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
CONFIG = {
    'csv_path': 'imu_range_data.csv',   # Input CSV path (change)
    'range_path': 'range_data.csv', # Optional separate range CSV (not used here)
    'output_csv': 'tight_ekf_output.csv',
    'col_time': 'timestamp(ns)',
    'col_ax': 'AX(m/s2)', 'col_ay': 'AY(m/s2)', 'col_az': 'AZ(m/s2)',
    'col_gx': 'GX(rad/s)', 'col_gy': 'GY(rad/s)', 'col_gz': 'GZ(rad/s)',
    'col_range_time': 'time',
    'col_range': 'dist_m',               # set to None if no range column
    'range_anchor': np.array([8.0, 0.5, 3.0]),
    'gyro_in_degrees': False,
    'init_samples': 200,                # samples used to compute initial accel mean
    # noise tuning (tweak for your sensor)
    'proc_noise_q_quat': 1e-6,          # small quaternion process noise
    'proc_noise_pos': 1e-4,
    'proc_noise_vel': 1e-3,
    'proc_noise_bg': 1e-8,
    'proc_noise_ba': 1e-6,
    'range_var': 0.1**2,               # variance of range measurement (m^2)
}
# ----------------------------------------

GRAVITY = np.array([0.0, 0.0, 9.80665])  # z-up; remove if your convention differs

# ----------------- helpers -----------------
def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_mul(q, r):
    # Hamilton product q * r, q,r = [w, x, y, z]
    w1,x1,y1,z1 = q
    w2,x2,y2,z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_from_omega(omega, dt):
    # omega: angular velocity vector [wx,wy,wz] (rad/s)
    # returns quaternion representing rotation during dt: q = [cos(theta/2), axis*sin(theta/2)]
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega / np.linalg.norm(omega)
    return np.concatenate(([np.cos(theta/2.0)], np.sin(theta/2.0)*axis))

def rotate_vec_by_quat(q, v):
    # rotate vector v by quaternion q (body->world if q is that mapping)
    qq = R.from_quat([q[1], q[2], q[3], q[0]])
    return qq.as_matrix().dot(v)

def state_pack(q, p, v, bg, ba):
    return np.hstack((q, p, v, bg, ba))

def state_unpack(x):
    q = x[0:4]; p = x[4:7]; v = x[7:10]; bg = x[10:13]; ba = x[13:16]
    return q, p, v, bg, ba

# numerical jacobian for discrete-time propagation: df/dx
def numerical_F(fun, x, imu, dt, eps=1e-6):
    n = x.size
    fx = fun(x, imu, dt)
    F = np.zeros((n,n))
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        xp = x + dx
        fxp = fun(xp, imu, dt)
        F[:, i] = (fxp - fx) / eps
    return F

# discrete propagation function: x_{k+1} = f(x_k, imu, dt)
def f_state(x, imu, dt):
    # imu: dict with 'gyro' (3,) rad/s, 'accel' (3,) m/s^2
    q, p, v, bg, ba = state_unpack(x)
    gyro = imu['gyro']
    accel = imu['accel']
    # remove bias
    omega = gyro - bg
    a_b = accel - ba
    # quaternion increment
    dq = quat_from_omega(omega, dt)
    q_next = quat_mul(q, dq)
    q_next = quat_normalize(q_next)
    # rotate accel to world and subtract gravity
    a_world = rotate_vec_by_quat(q_next, a_b) - GRAVITY
    # integrate velocity and position (simple constant accel over dt)
    v_next = v + a_world * dt
    p_next = p + v * dt + 0.5 * a_world * dt * dt
    # biases modeled as random walk (no change)
    bg_next = bg.copy()
    ba_next = ba.copy()
    return state_pack(q_next, p_next, v_next, bg_next, ba_next)

# measurement function for range (scalar)
def h_range(x, anchor):
    _, p, _, _, _ = state_unpack(x)
    return np.linalg.norm(p - anchor)

# analytical H for range measurement: dh/dx
def H_range(x, anchor):
    _, p, _, _, _ = state_unpack(x)
    dp = p - anchor
    r = np.linalg.norm(dp)
    if r < 1e-8:
        # near zero distance; return zeros
        H = np.zeros((1, x.size))
        return H
    H = np.zeros((1, x.size))
    H[0, 4:7] = (dp / r).T
    return H

# ---------------- EKF class ----------------
class TightCoupledEKF:
    def __init__(self, anchor, dt0, config):
        self.anchor = np.array(anchor)
        self.dt = dt0
        # state
        self.x = np.zeros(16)
        # covariance
        self.P = np.eye(16) * 1e-2
        # process noise matrix Q (discrete)
        qqq = config['proc_noise_q_quat']
        qpos = config['proc_noise_pos']
        qvel = config['proc_noise_vel']
        qbg = config['proc_noise_bg']
        qba = config['proc_noise_ba']
        # note: quaternion part gets small noise to avoid numerical singularities
        self.Q = np.diag(np.hstack((
            np.ones(4)*qqq, np.ones(3)*qpos, np.ones(3)*qvel, np.ones(3)*qbg, np.ones(3)*qba
        )))
        self.R_range = config['range_var']
        # small epsilon for numerical Jacobian
        self.eps_jac = 1e-6

    def init_state(self, q0, p0=None, v0=None, bg0=None, ba0=None):
        if p0 is None: p0 = np.zeros(3)
        if v0 is None: v0 = np.zeros(3)
        if bg0 is None: bg0 = np.zeros(3)
        if ba0 is None: ba0 = np.zeros(3)
        self.x = state_pack(quat_normalize(q0), p0, v0, bg0, ba0)

    def predict(self, imu, dt):
        self.dt = dt
        # discrete propagation
        f = lambda x, imu, dt: f_state(x, imu, dt)
        x_pred = f(self.x, imu, dt)
        # numerical Jacobian
        F = numerical_F(f, self.x, imu, dt, eps=self.eps_jac)
        # discrete covariance update
        self.P = F.dot(self.P).dot(F.T) + self.Q
        self.x = x_pred
        # normalize quaternion
        q, p, v, bg, ba = state_unpack(self.x)
        self.x[0:4] = quat_normalize(q)

    def update_range(self, r_meas):
        # compute measurement pred
        r_pred = h_range(self.x, self.anchor)
        # measurement Jacobian
        H = H_range(self.x, self.anchor)
        S = H.dot(self.P).dot(H.T) + self.R_range
        K = self.P.dot(H.T) / S  # (16x1)
        innov = r_meas - r_pred
        self.x = self.x + (K.flatten() * innov)
        # quaternion normalization after additive update: better to re-normalize q
        self.x[0:4] = quat_normalize(self.x[0:4])
        I = np.eye(self.P.shape[0])
        self.P = (I - K.dot(H)).dot(self.P)
        return r_pred

# ---------------- main pipeline ----------------
def run_tight_ekf(df_imu, df_range, config):
    time_col = config['col_time']
    ax = df_imu[config['col_ax']].to_numpy()
    ay = df_imu[config['col_ay']].to_numpy()
    az = df_imu[config['col_az']].to_numpy()
    gx = df_imu[config['col_gx']].to_numpy()
    gy = df_imu[config['col_gy']].to_numpy()
    gz = df_imu[config['col_gz']].to_numpy()
    times = df_imu[time_col].to_numpy()*1e-9

    rng_times = df_range["time"].to_numpy() * 1e-3  # convert ms to s
    rng = df_range["dist_m"].to_numpy()

    if config['gyro_in_degrees']:
        gx = np.deg2rad(gx); gy = np.deg2rad(gy); gz = np.deg2rad(gz)

    n = len(times)
    dt_arr = np.diff(times)
    dt_arr = np.concatenate(([dt_arr[0] if len(dt_arr)>0 else 0.01], dt_arr))

    # initialize quaternion from average accel (assume stationary initially)
    init_n = min(config['init_samples'], n)
    acc0 = np.array([np.mean(ax[:init_n]), np.mean(ay[:init_n]), np.mean(az[:init_n])])
    if np.linalg.norm(acc0) < 1e-8:
        q0 = np.array([0., 0., 0., 1.])
    else:
        acc0u = acc0 / np.linalg.norm(acc0)
        # want rotation that maps body accel direction to world -z (gravity)
        v1 = acc0u
        v2 = np.array([0.0, 0.0, -1.0])
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot)
        if abs(angle) < 1e-8:
            q0 = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = np.cross(v1, v2)
            axis = axis / np.linalg.norm(axis)
            q0 = np.concatenate(([np.cos(angle/2.0)], np.sin(angle/2.0)*axis))

    # initial position: if first range present, place on x-axis at that distance; else zero
    p0 = np.array([12,0.5,0])

    if not np.isnan(rng[0]):
        p0 = config['range_anchor'] + np.array([rng[0], 0.0, 0.0])
    v0 = np.zeros(3)
    bg0 = np.zeros(3)
    ba0 = np.zeros(3)

    ekf = TightCoupledEKF(anchor=config['range_anchor'], dt0=dt_arr[0], config=config)
    ekf.init_state(q0, p0, v0, bg0, ba0)

    # storage
    out = np.zeros((len(rng_times), 13))  # time, px,py,pz, vx,vy,vz, bgx,bgy,bgz, bax,bay,baz
    rpreds = np.full(len(rng_times), np.nan)

    range_cnt = 0
    while rng_times[range_cnt] < times[0]:
        range_cnt += 1

    for k in range(n):
        dt = dt_arr[k]
        imu = {'gyro': np.array([gx[k], gy[k], gz[k]]), 'accel': np.array([ax[k], ay[k], az[k]])}
        ekf.predict(imu, dt)

        while times[k] >= rng_times[range_cnt]:
            r_meas = rng[range_cnt]
            r_pred = np.nan
            if not np.isnan(r_meas):
                r_pred = ekf.update_range(r_meas)
            rpreds[range_cnt] = r_pred
            # store
            q, p, v, bg, ba = state_unpack(ekf.x)
            out[range_cnt, 0] = times[k]
            out[range_cnt, 1:4] = p
            out[range_cnt, 4:7] = v
            out[range_cnt, 7:10] = bg
            out[range_cnt, 10:13] = ba
            range_cnt += 1
            if range_cnt >= len(rng_times):
                break
        if range_cnt >= len(rng_times):
            break


        # store


    cols = ['time','px','py','pz','vx','vy','vz','bgx','bgy','bgz','bax','bay','baz']
    out_df = pd.DataFrame(out, columns=cols)
    out_df['r_meas'] = rng
    out_df['r_pred'] = rpreds
    out_df['speed'] = np.linalg.norm(out_df[['vx','vy','vz']].to_numpy(), axis=1)
    return out_df

# -------------- example synthetic test (quick sanity) --------------
def synthetic_demo():
    # simple 2D accel along x, constant accel, anchor at origin, small noise
    T = 8.0; fs = 100.0
    n = int(T*fs)+1
    t = np.linspace(0, T, n)
    # world accel (0.5 m/s^2 in x), gravity z applied to sensor reading
    a_world = np.vstack((0.5*np.ones(n), np.zeros(n), np.zeros(n))).T
    # sensor measures a_body = R^T * (a_world + gravity) ; assume no rotation (body==world)
    ax = a_world[:,0]; ay = a_world[:,1]; az = a_world[:,2] + GRAVITY[2]
    gx = np.zeros(n); gy = np.zeros(n); gz = np.zeros(n)
    # true trajectory
    p0 = np.array([5.0, 0.0, 0.0])
    v0 = np.zeros(3)
    p = p0 + np.outer(0.5*0.5*t*t, np.array([1.0,0.0,0.0]))  # actually p = p0 + 0.5*a*t^2
    # ranges to origin
    anchor = np.array([0.0,0.0,0.0])
    ranges = np.linalg.norm(p - anchor, axis=1) + np.random.normal(scale=0.03, size=n)

    df = pd.DataFrame({
        'time': t, 'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'range': ranges
    })
    res = run_tight_ekf(df, CONFIG)
    print(res.head())
    res.to_csv('synthetic_tight_output.csv', index=False)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(res['time'], res['px'], label='px'); plt.plot(res['time'], res['py'], label='py'); plt.legend(); plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(res['time'], res['speed']); plt.grid(True); plt.xlabel('time [s]'); plt.ylabel('speed [m/s]')
    plt.show()

# ---------------- main ----------------
if __name__ == '__main__':
    df_imu = pd.read_csv("../ESP_code/data/Long Experiment noon VIO/data_imu.csv")
    df_range = pd.read_csv("../ESP_code/data/Long Experiment noon Range/range_16_7.csv")
    gt_time_0 = 2627.448  # s
    vio_time_0 = 364.598  # s
    dt = 20.433  # s
    delta_t = gt_time_0 - vio_time_0 + dt
    df_imu["timestamp(ns)"] = df_imu["timestamp(ns)"] + delta_t * 1e9

    # basic checks


    res = run_tight_ekf(df_imu, df_range, CONFIG)
    res.to_csv(CONFIG['output_csv'], index=False)
    print(f"Saved results to {CONFIG['output_csv']}")
    # quick plots
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(res['time'], res['px'], label='px'); plt.plot(res['time'], res['py'], label='py'); plt.legend(); plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(res['time'], res['speed']); plt.grid(True); plt.xlabel('time [s]'); plt.ylabel('speed [m/s]')
    plt.tight_layout()
    plt.show()