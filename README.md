# POSEIDON FAIR Enrichment

This repository contains Python utilities for processing ESP/UWB experiment logs, estimating tag trajectories from UWB ranges, and comparing or streaming those trajectories with IMU and VIO data. The current workflows are mostly captured as `unittest` methods that act as scripts.

The main processing code is in `pos_Code/Experimental_Setup/Experimental_Setup.py`. The Aerolytics experiment examples are in `pos_Code/Experimental_Setup/test_Experimental_Setup.py`.

## Repository Layout

- `pos_Code/Experimental_Setup/`: experiment loader, UWB range filtering, multilateration, GT saving/loading, VIO/IMU loading, and stream/export helpers.
- `pos_Code/ESP_code/`: ESP UDP logging and CSV folder reader. Experiment data is expected under `pos_Code/ESP_code/data/`.
- `pos_Code/Adapted_UPF/` and `pos_Code/Adapted_UPF_test/`: particle-filter and simulation-related code.
- `Test_cases/`: older or broader RPE/Aerolytics test workflows.
- `Video/`: video helper scripts.
- `uwb_firmware/` and `pos_Code/ESP32C3_code/`: firmware-related code and sketches.

## Data Layout

The Aerolytics data path used by the current scripts is:

```text
pos_Code/ESP_code/data/
  Long Experiment noon/           # raw ESP UDP CSV files: udp_data_1.csv, ...
  Long Experiment noon VIO/
    data.csv                      # VIO pose/velocity/gravity/gyro output
    data_imu.csv                  # high-rate IMU output from the VIO device
  Aerolytics_Long_GT_3/
    tag_16_gt.csv                 # saved UWB-derived ground truth
    tag_16_gt.pkl
```

The raw ESP CSV rows are variable width. Each row starts with device/time/IMU fields, followed by `valid_ranges` range blocks. Because of that, the repo reads these files with `csv.reader` rather than `pandas.read_csv`.

## Aerolytics Ground Truth Pipeline

The Aerolytics ground-truth track estimates the location of moving tag `16` from UWB distances to fixed anchors.

1. `load_cal_aero_setup()` defines calibrated anchor positions for anchor IDs `0..15`.
2. `Experiment.load_data("../ESP_code/data/Long Experiment noon")` reads all `udp_data_N.csv` files and parses each row into:
   - `odom_data`: one row per ESP packet, including time, ID, IMU-like fields, and anchor position if known.
   - `range_data`: one row per UWB measurement, with `time`, sender `id`, receiver `rid`, `dist_m`, `fp_rssi`, and `rx_rssi`.
3. `set_anchors()` marks known anchor IDs and writes their calibrated positions into `odom_data`.
4. `calculate_gt_trajectory_of_tag(16, time_horizon=0.1, max_std=0.1)` filters ranges to each anchor and estimates tag position over time.
5. `calculate_position()` keeps recent ranges from the last second and only solves when at least five anchor ranges are available.
6. `calculate_3D_NLS()` solves nonlinear least squares multilateration with a robust `soft_l1` loss. It also estimates covariance from the solver Jacobian.
7. `save_gt()` writes `tag_16_gt.csv` and `tag_16_gt.pkl`.

Example:

```python
from pos_Code.Experimental_Setup.test_Experimental_Setup import load_cal_aero_setup

exp_setup, _ = load_cal_aero_setup(
    debug=True,
    folder_path="../ESP_code/data/Long Experiment noon",
)
exp_setup.calculate_gt_trajectory_of_tag(16, time_horizon=0.1, max_std=0.1)
exp_setup.save_gt("../ESP_code/data/Aerolytics_Long_GT_3")
```

The saved ground-truth table has columns:

```text
time, id, px, py, pz, cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz
```

## VIO Data

The VIO file `Long Experiment noon VIO/data.csv` is loaded with:

```python
exp_setup.load_vio_data("../ESP_code/data/Long Experiment noon VIO/data.csv", 16)
```

This only adds an `id` column and stores the rows as `exp_setup.vio_data`.

Alignment to the UWB/Aerolytics frame is manual. The current examples use a translation, quaternion, and time offset:

```python
t = [12.63509959, 0.94427232, 1.59824491]
q = [0.29832535, -0.73748568, 0.60547778, -0.02266103]
delta_t = 2627.448 - 364.598 + 20.433

exp_setup.set_vio_transformation(t, q, delta_t, 16)
```

`set_vio_transformation()` rotates VIO positions by `q`, translates them by `t`, and shifts VIO timestamps by `delta_t`. VIO is not used to create the saved UWB-derived ground truth; it is used for plotting, comparison, and extracting motion-related signals.

## IMU Data

There are two IMU-like sources:

- The raw ESP packets in `Long Experiment noon/`, parsed into `odom_data`.
- The VIO-device IMU file `Long Experiment noon VIO/data_imu.csv`, loaded with `load_imu_data()`.

`load_imu_data()` converts VIO IMU timestamps from nanoseconds to milliseconds, applies the same time offset as VIO, maps accelerometer and gyroscope columns into `odom_data`, and replaces existing rows for the given ID.

Example:

```python
filepath = "../ESP_code/data/Long Experiment noon VIO/data_imu.csv"
delta_t = 2627.448 - 364.598 + 20.433

exp_setup.load_imu_data(filepath, 16, t_diff=delta_t)
```

The IMU data is used by `stream_odom_from_gt(..., imu_bool=True)` and `stream_exp_id()`. These functions derive a state from a sliding ground-truth window:

- position: mean GT position in the window,
- velocity: first-to-last GT displacement divided by window duration,
- heading: `atan2(dy, dx)`, assuming mostly forward motion,
- angular velocity: heading difference over time,
- acceleration and gyro: average IMU samples between the previous and current GT time.

`stream_exp_id()` combines this state and covariance with filtered UWB range vectors. The example export in `test_stream_exp_id()` writes a flat CSV called `Aero_full_exp.csv`.

## Useful Workflows

Plot Aerolytics ground truth and anchors:

```python
exp_setup, anchors = load_cal_aero_setup(debug=True, folder_path=None)
exp_setup.load_gt("../ESP_code/data/Aerolytics_Long_GT_3")
exp_setup.set_anchors(anchors)
exp_setup.plot_3D()
```

Load and align VIO for comparison:

```python
exp_setup, _ = load_cal_aero_setup()
exp_setup.load_gt("../ESP_code/data/Aerolytics_Long_GT_2")
exp_setup.load_vio_data("../ESP_code/data/Long Experiment noon VIO/data.csv", 16)
exp_setup.set_vio_transformation(t, q, delta_t, 16)
exp_setup.plot_3D()
```

Create a flat Aerolytics stream with ranges, GT-derived state, covariance, and IMU averages:

```python
exp_setup, anchors = load_cal_aero_setup(debug=True)
exp_setup.load_gt("../ESP_code/data/Aerolytics_Long_GT_3")
exp_setup.set_anchors(anchors)
exp_setup.load_imu_data("../ESP_code/data/Long Experiment noon VIO/data_imu.csv", 16, t_diff=delta_t)

rows = []
for data in exp_setup.stream_exp_id(16, history=1):
    rows.append(data)
```

## Known Caveats

- The repo does not currently include a dependency manifest such as `requirements.txt` or `pyproject.toml`.
- Many workflows live inside `unittest` methods, but they are often used as scripts rather than automated tests.
- `get_vio_transform()` reads velocity and then overwrites it with gravity vectors before returning, so its returned first array is gravity, not VIO linear velocity.
- `load_imu_data()` maps `GX` into `gy` and `GY` into `gx`; this may be an intended frame conversion, but it is not documented in code.
- The GT-derived angular velocity path may leave `w` as `nan` because `previous_theta` is initialized as `nan` and is only updated in the non-NaN branch.
- The Aerolytics GT is UWB-derived multilateration. It should be treated as processed ground truth for this workflow, not as an independent motion-capture source.

## Development Notes

Run scripts from locations that keep the relative data paths valid. Most Aerolytics examples assume they are executed from `pos_Code/Experimental_Setup/` because they reference data paths like `../ESP_code/data/...`.

Before changing processing behavior, inspect the relevant test workflow and the data folder it references. The code has several hard-coded experiment-specific constants for anchor positions, time alignment, and tag IDs.
