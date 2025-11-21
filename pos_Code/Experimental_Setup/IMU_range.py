"""
imu_range_ekf.py

Complete, standalone Python script that:
- reads a CSV with timestamps, accel (ax,ay,az), gyro (gx,gy,gz), and optional range measurements to a fixed anchor
- runs an orientation estimator (Madgwick) to produce body->world rotation
- runs an EKF that estimates position, velocity, and accelerometer bias, fusing IMU propagation and range corrections
- writes results (time, px,py,pz, vx,vy,vz, speed) to a CSV and optionally plots them

How to use:
- Edit the CSV path and column names in the `CONFIG` section below to match your data.
- Make sure timestamps are in seconds and accelerometer in m/s^2, gyro in rad/s (script converts deg->rad if needed).
- If your range data is sparse, put NaN for missing range entries.

Dependencies: numpy, scipy, pandas, matplotlib

This script is intentionally self-contained and simple to adapt. It uses a compact Madgwick implementation (no external libs).

Author: ChatGPT (GPT-5 Thinking mini)
"""

import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# -------------------------------
# CONFIG: change to match your CSV
# -------------------------------
CONFIG = {
    'csv_path': 'imu_range_data.csv',  # input CSV
    'output_csv': 'imu_range_ekf_output.csv',
    # expected column names (case-sensitive):
    'col_time': 'time',   # seconds
    'col_ax': 'AX(m/s2)',
    'col_ay': 'AY(m/s2)',
    'col_az': 'AZ(m/s2)',
    'col_gx': 'GX(rad/s)',
    'col_gy': 'GY(rad/s)',
    'col_gz': 'GZ(rad/s)',
    'col_range': 'range',  # set to None if no range column
    'range_anchor': [0.0, 0.0, 0.0],  # anchor position in world frame
    'gyro_in_degrees': False,  # set True if gyros in deg/s
    'assume_first_range_for_init': True,  # use first range to initialize position along x-axis
}

# -------------------------------
# Utility functions: quaternions
# -------------------------------

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_mul(q1, q2):
    # Hamilton product, q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z])

def quat_rotate(q, v):
    # rotate vector v (3,) by quaternion q (w,x,y,z)
    qv = np.concatenate(([0.0], v))
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]

# -------------------------------
# Madgwick filter (compact)
# -------------------------------
class MadgwickAHRS:
    def __init__(self, sample_period=1/256.0, beta=0.1):
        self.sample_period = sample_period
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update_imu(self, gyro, accel):
        # gyro: rad/s (3,), accel: m/s^2 (3,)
        q1, q2, q3, q4 = self.q
        ax, ay, az = accel
        gx, gy, gz = gyro

        # normalize accel
        norm_a = np.linalg.norm([ax, ay, az])
        if norm_a == 0:
            return
        ax, ay, az = ax / norm_a, ay / norm_a, az / norm_a

        # objective function and Jacobian (from Madgwick paper)
        f1 = 2*(q2*q4 - q1*q3) - ax
        f2 = 2*(q1*q2 + q3*q4) - ay
        f3 = 2*(0.5 - q2*q2 - q3*q3) - az
        J_11or24 = 2*q3
        J_12or23 = 2*q4
        J_13or22 = 2*q1
        J_14or21 = 2*q2
        J_32 = 2*J_14or21
        J_33 = 2*J_11or24

        # gradient
        grad = np.array([
            J_14or21*f2 - J_11or24*f1,
            J_12or23*f1 + J_13or22*f2 - J_32*f3,
            J_12or23*f2 - J_33*f3 - J_13or22*f1,
            J_14or21*f1 + J_11or24*f2
        ])
        grad = grad / np.linalg.norm(grad)

        # quaternion rate from gyro
        q_dot_omega = 0.5 * quat_mul(self.q, np.array([0.0, gx, gy, gz]))

        # integrate
        q_dot = q_dot_omega - self.beta * grad
        self.q = self.q + q_dot * self.sample_period
        self.q = quat_normalize(self.q)

    def get_rotation_matrix(self):
        # returns 3x3 rotation matrix body->world
        # use scipy Rotation for clarity
        q = self.q
        return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

# -------------------------------
# EKF for [p, v, ba]
# -------------------------------
class IMURangeEKF:
    def __init__(self, anchor_pos, dt, process_noise=None, range_var=0.1**2):
        self.dt = dt
        self.anchor = np.array(anchor_pos)
        # state: p(3), v(3), ba(3)
        self.x = np.zeros(10)
        self.P = np.eye(10) * 1.0

        # process noise: set reasonable defaults if None
        if process_noise is None:
            q_pos = 1e-4
            q_vel = 1e-3
            q_ba = 1e-6
            self.Q = np.diag(np.hstack((np.ones(3)*q_pos, np.ones(3)*q_vel, np.ones(3)*q_ba)))
        else:
            self.Q = process_noise

        self.R = range_var

    def init_position_from_range(self, r0):
        # naive init: place on x-axis at distance r0 from anchor
        self.x[0:3] = self.anchor + np.array([r0, 0.0, 0.0])

    def predict(self, a_m, Rbw):
        # a_m: raw accel body frame (3,), Rbw: body->world rotation (3x3)
        p = self.x[0:3].copy()
        theta  = self.x[3].copy()
        v = self.x[3:6].copy()
        ba = self.x[6:9].copy()

        # acceleration in world (note gravity as +z up 9.80665)
        g = np.array([0.0, 0.0, 9.80665])
        a_world = Rbw.dot(a_m - ba) + g

        # propagate state (constant accel during dt)
        p_pred = p + v*self.dt + 0.5 * a_world * self.dt*self.dt
        v_pred = v + a_world * self.dt
        ba_pred = ba

        self.x[0:3] = p_pred
        self.x[3:6] = v_pred
        self.x[6:9] = ba_pred

        # build F
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[0:3, 6:9] = -0.5 * Rbw * (self.dt*self.dt)
        F[3:6, 6:9] = - Rbw * self.dt

        self.P = F.dot(self.P).dot(F.T) + self.Q

    def update_range(self, r_meas):
        p = self.x[0:3]
        dp = p - self.anchor
        r_pred = np.linalg.norm(dp)
        if r_pred < 1e-8:
            return r_pred

        H = np.zeros((1,9))
        H[0, 0:3] = (dp / r_pred).T

        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T) / S  # (9x1)

        innov = r_meas - r_pred
        self.x = self.x + (K.flatten() * innov)
        self.P = (np.eye(9) - K.dot(H)).dot(self.P)
        return r_pred

# -------------------------------
# Main pipeline
# -------------------------------

def run_pipeline(df_imu, df_range, config):
    # prepare
    t_col = config['col_time']
    axc = config['col_ax']; ayc = config['col_ay']; azc = config['col_az']
    gxc = config['col_gx']; gyc = config['col_gy']; gzc = config['col_gz']

    times = df_imu["timestamp(ns)"].to_numpy()*1e-9  # convert ns to s
    ax = df_imu[axc].to_numpy(); ay = df_imu[ayc].to_numpy(); az = df_imu[azc].to_numpy()
    gx = df_imu[gxc].to_numpy(); gy = df_imu[gyc].to_numpy(); gz = df_imu[gzc].to_numpy()

    rng_times = df_range["time"].to_numpy()*1e-3  # convert ms to s
    rng = df_range["dist_m"].to_numpy()
    # convert gyro to rad/s if needed
    if config['gyro_in_degrees']:
        gx = np.deg2rad(gx); gy = np.deg2rad(gy); gz = np.deg2rad(gz)

    n = len(times)
    # compute dt array (use first dt for Madgwick initialization)
    dt_arr = np.diff(times)
    dt_arr = np.concatenate(([dt_arr[0] if len(dt_arr)>0 else 0.01], dt_arr))
    dt_ekf = np.diff(rng_times)
    dt_ekf = np.concatenate(([dt_ekf[0] if len(dt_ekf)>0 else 0.1], dt_arr))
    # create filters
    madgwick = MadgwickAHRS(sample_period=dt_arr[0], beta=0.12)
    ekf = IMURangeEKF(anchor_pos=config['range_anchor'], dt=dt_ekf[0])

    # initialize using first range if available
    if config['assume_first_range_for_init'] and not np.isnan(rng[0]):
        ekf.init_position_from_range(rng[0])
    ekf.x[self.x[0:3]] = [4,0]
    # allocate output
    out = np.zeros((len(rng_times), 10))  # t, px,py,pz, vx,vy,vz, speed, r_meas, r_pred

    prev_a_world = np.zeros(3)

    range_cnt = 0
    while rng_times[range_cnt] < times[0]:
        range_cnt += 1

    for k in range(n):
        dt = dt_arr[k]
        madgwick.sample_period = dt
        ekf.dt = dt

        a_m = np.array([ax[k], ay[k], az[k]])
        gyro = np.array([gx[k], gy[k], gz[k]])

        # orientation update
        madgwick.update_imu(gyro, a_m)
        Rbw = madgwick.get_rotation_matrix()

        # predict
        ekf.predict(a_m, Rbw)

        # optional range update
        while times[k] >= rng_times[range_cnt]:
            r_meas = rng[range_cnt]
            r_pred = np.nan
            if not np.isnan(r_meas):
                r_pred = ekf.update_range(r_meas)
            # store
            p = ekf.x[0:3].copy()
            v = ekf.x[3:6].copy()
            speed = np.linalg.norm(v)
            out[range_cnt] = np.hstack((times[k], p, v, speed, r_meas if not np.isnan(r_meas) else np.nan, r_pred if r_pred is not None else np.nan))
            range_cnt += 1
            if range_cnt >= len(rng_times):
                break
        if range_cnt >= len(rng_times):
            break
    cols = ['time','px','py','pz','vx','vy','vz','speed','r_meas','r_pred']
    out_df = pd.DataFrame(out, columns=cols)
    return out_df

# -------------------------------
# Helper: example synthetic test (optional)
# -------------------------------
def synthetic_test():
    # Creates a short synthetic dataset to sanity-check the pipeline
    T = 10.0
    fs = 100.0
    n = int(T*fs)+1
    t = np.linspace(0, T, n)

    # simulate simple motion: constant acceleration in world x
    accel_world = np.vstack((np.ones(n)*0.5, np.zeros(n), np.zeros(n))).T  # 0.5 m/s^2 along x
    # assume no rotation (body==world)
    ax = accel_world[:,0]
    ay = accel_world[:,1]
    az = accel_world[:,2] - 9.80665  # sensor measures gravity + lin accel in body

    # gyros zero
    gx = np.zeros(n); gy = np.zeros(n); gz = np.zeros(n)

    # anchor at origin, initial p = [5,0,0], ranges = ||p - anchor||
    p0 = np.array([5., 0., 0.])
    v0 = np.array([0., 0., 0.])
    p = p0 + v0*t.reshape(-1,1) + 0.5*(accel_world)*(t.reshape(-1,1)**2)
    anchor = np.array([0.,0.,0.])
    ranges = np.linalg.norm(p - anchor, axis=1) + np.random.normal(scale=0.05, size=n)

    df = pd.DataFrame({
        'time': t,
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'range': ranges
    })

    result = run_pipeline(df, CONFIG)
    print(result.head())
    result.to_csv('synthetic_output.csv', index=False)
    # plot speed
    plt.figure()
    plt.plot(result['time'], result['speed'])
    plt.xlabel('time [s]')
    plt.ylabel('speed [m/s]')
    plt.title('Estimated speed (synthetic test)')
    plt.grid(True)
    plt.show()

# -------------------------------
# Main: read CSV, run, save
# -------------------------------
if __name__ == '__main__':

    df_imu = pd.read_csv("../ESP_code/data/Long Experiment noon VIO/data_imu.csv")
    df_range = pd.read_csv("../ESP_code/data/Long Experiment noon Range/range_16_7.csv")
    gt_time_0 = 2627.448  # s
    vio_time_0 = 364.598  # s
    dt = 20.433  # s
    delta_t = gt_time_0 - vio_time_0 + dt
    df_imu["timestamp(+ns)"] = df_imu["timestamp(ns)"] + delta_t * 1e9

    # basic checks

    out_df = run_pipeline(df_imu, df_range, CONFIG)
    out_df.to_csv(CONFIG['output_csv'], index=False)
    print(f"Saved EKF output to {CONFIG['output_csv']}")

    # quick plots
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(out_df['time'], out_df['px'], label='px')
    plt.plot(out_df['time'], out_df['py'], label='py')
    plt.plot(out_df['time'], out_df['pz'], label='pz')
    plt.legend(); plt.ylabel('position [m]'); plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(out_df['time'], out_df['speed'])
    plt.ylabel('speed [m/s]'); plt.xlabel('time [s]'); plt.grid(True)
    plt.tight_layout()
    plt.show()
