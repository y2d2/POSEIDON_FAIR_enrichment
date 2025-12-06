import deprecation
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from pos_Code.ESP_code.ESP_Class import ESP_wifi_module
from scipy.optimize import minimize, least_squares


class Experiment():
    def __init__(self, debug=False):
        self.odom_columns = ["time", "id", "mag_x", "mag_y", "mag_z", "T",
                            "gx", "gy", "gz", "ax", "ay", "az", "px", "py", "pz", "Anchor"]
        self.range_columns = ["time", "id", "rid", "dist_m", "fp_rssi", "rx_rssi"]
        self.odom_data = pd.DataFrame(columns=self.odom_columns)
        self.range_data = pd.DataFrame(columns=self.range_columns)
        self.tag_gts_columns = ["time", "id", "px", "py", "pz", "cov_xx", "cov_xy", "cov_xz", "cov_yy", "cov_yz", "cov_zz"]
        self.tag_gts = pd.DataFrame(columns=self.tag_gts_columns)
        self.debug = debug

    #################################################################################
    #### LOAD DATA
    #################################################################################
    def load_data(self, folder_path):
        esp_wifi = ESP_wifi_module()
        data = esp_wifi.read_folder(folder_path)
        odom_data = []
        range_data = []
        for i, row in enumerate(data):
            print(f"Processing {(i+1)*100/len(data)}%", end="\r")
            if self.debug:
                print(f"Processing {(i+1)*100/len(data)}%")
            odom_row, range_rows = self.parse_row(row)
            odom_data.append(odom_row)
            range_data.extend(range_rows)
        self.odom_data = pd.DataFrame(odom_data, columns=self.odom_columns)
        self.range_data = pd.DataFrame(range_data, columns=self.range_columns)
        print(f"Loaded {len(self.odom_data)} odom entries and {len(self.range_data)} range entries.")

    def parse_row(self, row):
        float_row =  [float(value) for value in row]
        sid =  int(float_row[0])

        t = float_row[1]
        magx, magy, magz = float_row[2:5]
        temp = float_row[5]
        gx, gy, gz = float_row[6:9]
        ax, ay, az = float_row[9:12]
        imu_data = [t, sid, magx, magy, magz, temp, gx, gy, gz, ax, ay, az, 0, 0, 0, False]

        valid_ranges = int(float_row[12])
        ranges = []
        for i in range(valid_ranges):
            rid = int(float_row[13 + i * 4])
            dist = float_row[14 + i * 4]
            fp_rssi = int(float_row[15 + i * 4])
            rx_rssi = int(float_row[16 + i * 4])
            range_data = [t, sid, rid, dist, fp_rssi, rx_rssi]
            ranges.append(range_data)
        return imu_data, ranges

    #################################################################################
    #### DATA CHECKS
    #################################################################################
    def check_frequencies(self):
        for id in self.odom_data['id'].unique():
            device_data = self.odom_data[self.odom_data['id'] == id].sort_values(by='time')
            time_diffs = device_data['time'].diff().dropna()/1000
            mean_freq = 1 / time_diffs.mean()
            std_freq = time_diffs.std() / (time_diffs.mean() ** 2)
            print(f"Device {id}: Mean Frequency = {mean_freq:.2f} Hz, Std Dev = {std_freq:.4f} Hz")


    #################################################################################
    #### ANCHOR SETUP
    #################################################################################

    def set_anchors(self, anchor_ids):
        #in self.odom_data set the pos of the anchors
        for anchor_id in anchor_ids:
            anchor_val = [anchor_ids[anchor_id][0], anchor_ids[anchor_id][1], anchor_ids[anchor_id][2], True]
            if int(anchor_id) in self.odom_data['id'].values:
                if self.debug:
                    print(f"Setting anchor {anchor_id} position to {anchor_val}")
                self.odom_data.loc[self.odom_data['id'] == int(anchor_id), ['px', 'py', 'pz', 'Anchor']] = anchor_val
            else:
                print(f"Warning: Anchor ID {anchor_id} not found in data.")

    def get_anchor_ranges_statistics(self):
        odom_data_anchors = self.odom_data[self.odom_data['Anchor'] == True]
        unique_ids = odom_data_anchors['id'].unique()
        mean_matrix = np.full((25, 25), np.nan)
        std_matrix = np.full((25, 25), np.nan)
        for i in range(25):
            for j in range(i+1, 25):
                if i in unique_ids and j in unique_ids:
                    data = self.range_data[(self.range_data['id'].isin([i, j]))
                                           & (self.range_data['rid'].isin([j, i]))]
                    if not data.empty:
                        mean_dist = data['dist_m'].mean()
                        std_dist = data['dist_m'].std()
                        mean_matrix[i, j] = mean_dist
                        std_matrix[i, j] = std_dist
                        # print(f"Range {i} to {j}: Mean = {mean_dist:.2f} m, Std = {std_dist:.2f} m, Samples = {len(data)}")
        return mean_matrix, std_matrix

    def calibrate_anchor_pos(self, bounds=0, bounds_list =None, plot=False):
        def calculate_error(positions_flat, ordered_ids, mean_dis_matrix, std_dis_matrix):
            positions = positions_flat.reshape((len(ordered_ids), 3))
            dis_matrix = np.full((25, 25), np.nan)
            used_id = []
            for i, sid in enumerate(ordered_ids):
                used_id.append(sid)
                for j, rid in enumerate(ordered_ids):
                    if rid not in used_id:
                        pos_i = positions[i]
                        pos_j = positions[j]
                        dist_ij = np.linalg.norm(pos_i - pos_j)
                        dis_matrix[sid,rid] = dist_ij
                        # print(f"Initial distance between {i} and {j}: {dist_ij:.2f} m")
            error_matrix = mean_dis_matrix - dis_matrix
            #mask error matrix if std_dis_matrix > 0.5:
            error_matrix[std_dis_matrix > 0.1] = np.nan
            print(f"Current error: {np.nansum(np.abs(error_matrix))}")
            return np.nansum(np.abs(error_matrix))

        def create_bounds(init_pos, bound):
            bound_list = []
            for pos in init_pos:
                bound_list.append((pos[0]-bound, pos[0]+bound))
                bound_list.append((pos[1]-bound, pos[1]+bound))
                if pos[2] == 3:
                    bound_list.append((3-bound,3+bound))
                else:
                    bound_list.append((pos[2]-bound, pos[2]+bound))
            return bound_list

        mean_dis_matrix, std_dis_matrix = self.get_anchor_ranges_statistics()
        init_positions = self.odom_data[self.odom_data['Anchor'] == True][
            ['id', 'px', 'py', 'pz']].drop_duplicates().set_index('id').to_dict('index')
        flattened_init_pos = []
        ordered_ids = []
        for i in range(25):
            if i in init_positions:
                flattened_init_pos.extend([init_positions[i]['px'], init_positions[i]['py'], init_positions[i]['pz']])
                ordered_ids.append(i)

        flattened_pos = np.array(flattened_init_pos)
        if bounds_list is None:
            bounds_list = create_bounds(np.array(flattened_init_pos).reshape((-1,3)), bounds)
        result = minimize(
            calculate_error,
            flattened_pos,
            args=(ordered_ids, mean_dis_matrix, std_dis_matrix),
            method='L-BFGS-B', options={'disp': True},
            bounds=bounds_list
        )
        flatten_pos_3d = np.array(flattened_init_pos).reshape((-1,3))
        new_pos_3d = result.x.reshape((-1,3))
        print(flattened_init_pos)
        print((flattened_pos - result.x).reshape((-1,3)))

        print("[")
        for i in new_pos_3d:
            print(f"  [{i[0]:.3f}, {i[1]:.3f}, {i[2]:.3f}],")
        print("]")
        for i in ordered_ids:
            #change the pos in self.odom_data for the anchor
            idx = ordered_ids.index(i)
            new_pos = [result.x[idx*3], result.x[idx*3+1], result.x[idx*3+2]]
            self.odom_data.loc[self.odom_data['id'] == i, ['px', 'py', 'pz']] = new_pos
        if plot:
            ax_3d= plt.figure().add_subplot(projection='3d')
            ax_3d.scatter(flatten_pos_3d[:,0], flatten_pos_3d[:,1], flatten_pos_3d[:,2], c='b')
            ax_3d.scatter(new_pos_3d.reshape((-1,3))[:,0], new_pos_3d.reshape((-1,3))[:,1], new_pos_3d.reshape((-1,3))[:,2], c='r')
            plt.show()

    #################################################################################
    ####  GROUND TRUTH TRAJECTORY
    #################################################################################

    def filter_ranges(self, sid, rid, time_horizon, max_std):
        time_horizon= time_horizon*1e3 #convert to ms
        data = self.range_data[(self.range_data['id'].isin([rid, sid]))
                               & (self.range_data['rid'].isin([sid, rid]))]
        if data.empty:
            print(f"No range data between {sid} and {rid}")
            return pd.DataFrame(columns=self.range_columns)
        data = data.sort_values(by='time')
        filtered_data = []
        stds= []
        distances = []
        times = []
        for i, row in data.iterrows():
            times.append(row['time'])
            distances.append(row['dist_m'])
            std_dist = np.std(distances)
            stds.append(std_dist)
            if std_dist > max_std:
                distances.pop(-1)
                times.pop(-1)
            elif len(distances ) >= int(time_horizon/1e3*10*0.9): # at least 90% of 10hz.
                filtered_data.append(row)
            while row['time'] - times[0] > time_horizon:
                times.pop(0)
                distances.pop(0)
                if times == []:
                    break
            # t = row['time']
            # window = data[(data['time'] >= t - time_horizon) & (data['time'] <= t + time_horizon)]
            # if len(window) > int(time_horizon/1e3*10*0.9): # at least 90% of 10hz.
            #     mean_dist = window['dist_m'].mean()
            #     std_dist = window['dist_m'].std()
            #     stds.append(std_dist)
            #     if std_dist <= max_std:
        return pd.DataFrame(filtered_data, columns=self.range_columns), data, stds

    def filter_all_ranges(self, id, time_horizon, max_std):
        ids = self.range_data["id"].unique()
        filtered_ranges = pd.DataFrame(columns=self.range_columns)
        for rid in ids:
            if rid != id:
                range_data, _,_ = self.filter_ranges(id, rid, time_horizon, max_std)
                filtered_ranges = pd.concat([filtered_ranges, range_data], ignore_index=True)
        return filtered_ranges


    def calculate_gt_trajecotries(self, time_horizon=1, max_std=1.0, fixed_z=None):
        tag_ids = self.odom_data[self.odom_data['Anchor'] == False]['id'].unique()
        for sid in tag_ids:
            if self.debug:
                print(f"Calculating trajectory for tag {sid}")
            self.calculate_gt_trajectory_of_tag(sid, time_horizon, max_std,fixed_z )

    def calculate_gt_trajectory_of_tag(self, sid, time_horizon=1, max_std=1.0, fixed_z=None, frequency=None):
        eps = 0
        if frequency is not None:
            eps = 1e3 / frequency

        anchors_ids = self.odom_data[self.odom_data['Anchor'] == True]['id'].unique()
        ranges_df = pd.DataFrame(columns=self.range_columns)
        for rid in anchors_ids:
            if rid != sid:
                if self.debug:
                    print(f"Calculating ranges between {sid} and {rid}")
                range_data, original, stds = self.filter_ranges(sid, rid, time_horizon, max_std)
                ranges_df = pd.concat([ranges_df, range_data], ignore_index=True)
        # arrange by time
        ranges_df = ranges_df.sort_values(by='time')
        current_ranges = [np.array([idx, np.nan, np.nan, np.nan]) for idx in anchors_ids]
        current_time = None
        for i, row in ranges_df.sort_values(by='time').iterrows():
            if current_time is None:
                current_time = row['time']
            if current_time + eps < row['time']: # new time step every 100ms = 10hz
                current_time = row['time']
                pos, cov = self.calculate_position(current_ranges, current_time, sid, fixed_z)
                if self.debug:
                    print(f"Tag {sid} at time {current_time/1e3} s: Position = {pos}")
                self.tag_gts = pd.concat([self.tag_gts, pd.DataFrame([[current_time, sid, pos[0], pos[1], pos[2], cov[0,0], cov[0,1], cov[0,2], cov[1,1],cov[1,2], cov[2,2]]], columns=self.tag_gts_columns)])
            idx  = row['rid']
            if idx == sid:
                idx = row['id']
            # Find the index of the anchor in anchors_ids
            range_index = list(anchors_ids).index(idx)
            current_ranges[range_index] = np.array([int(idx), row['time'], row['dist_m'], row['fp_rssi']])

            # self.odom_data.loc[(self.odom_data['id'] == sid) & (self.odom_data['time'] == row['time'])

    def calculate_position(self, current_ranges, current_time, sid, fixed_z=None):
        # form https://www.researchgate.net/publication/224222701_Static_positioning_using_UWB_range_measurements/link/0fcfd51071298bfea5000000/download
        X = np.array([np.nan, np.nan, np.nan])
        cov = np.full((3,3), np.nan)
        # get latest measurement form tag_gts
        tag_gt = self.tag_gts[self.tag_gts['id'] == sid]
        if not tag_gt.empty:
            latest_gt = tag_gt[tag_gt['time'] <= current_time].sort_values(by='time').iloc[-1]
            X = np.array([latest_gt['px'], latest_gt['py'], latest_gt['pz']])


        usefull_ranges = []
        for r in current_ranges:
            if r[1] is not np.nan and r[1] > current_time - 1e3: # last second
                usefull_ranges.append(r)
        if len(usefull_ranges) >= 5:
            if fixed_z is not None:
                X, cov = self.calculate_2D_NLS(usefull_ranges, initial_guess=X, fixed_z=fixed_z)

            # X = self.calculate_3D_NLS(usefull_ranges)
            else:
                X, cov = self.calculate_3D_NLS(usefull_ranges, initial_guess=X)
            # X = np.linalg.solve(AtA, AtB)
            if self.debug:
                error = self.calculate_error_on_range(usefull_ranges, X)
                print(f"Errors {error}")
        return X, cov

    def calculate_3D_LS(self, usefull_ranges):
        A = np.empty((0, 3))
        b = np.empty((0,))
        for i, row in enumerate(usefull_ranges):
            xid = int(row[0])
            if i == 0:
                pos_0 = self.odom_data[(self.odom_data['id'] == xid) & (self.odom_data['Anchor'] == True)][
                    ['px', 'py', 'pz']].iloc[0].to_numpy()
                d0 = row[2]
            else:
                pos_i = self.odom_data[(self.odom_data['id'] == xid) & (self.odom_data['Anchor'] == True)][
                        ['px', 'py', 'pz']].iloc[0].to_numpy()
                di = row[2]
                Arow = 2 * (pos_i - pos_0)
                A = np.append(A, [Arow], axis=0)
                b_row = d0 ** 2 - di ** 2 + pos_i[0] ** 2 - pos_0[0] ** 2 + pos_i[1] ** 2 - pos_0[1] ** 2 + pos_i[2] ** 2 - pos_0[2] ** 2
                b = np.append(b, [b_row], axis=0)
        pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return pos

    def calculate_2D_NLS(self, usefull_ranges, initial_guess = None, fixed_z = 0):
        bounds = [(-20, -20), (40, 40)]
        anchors = []
        distances = []
        for i, row in enumerate(usefull_ranges):
            xid = int(row[0])
            anchors.append(self.odom_data[(self.odom_data['id'] == xid) & (self.odom_data['Anchor'] == True)][
                               ['px', 'py', 'pz']].iloc[0].to_numpy())
            distances.append(row[2])
        anchors = np.array(anchors)
        distances = np.array(distances)

        # If no initial guess, use centroid of anchors
        if np.any(np.isnan(initial_guess)):
            initial_guess = np.mean(anchors, axis=0)
        initial_guess = np.array([initial_guess[0], initial_guess[1]])

        if bounds is not None:
            lower, upper = np.array(bounds[0]), np.array(bounds[1])
            initial_guess = np.clip(initial_guess, lower, upper)

        def residuals(pos):
            # Difference between measured and computed distances
            aug_pos = np.array([pos[0], pos[1], fixed_z])
            return np.linalg.norm(anchors - aug_pos, axis=1) - distances

        # Solve nonlinear least squares
        result = least_squares(
            residuals,
            x0=initial_guess,
            bounds=bounds,
            method='trf',  # Trust Region Reflective algorithm (good with bounds)
            loss='soft_l1',  # robust to outliers
            f_scale=0.5,
            verbose=self.debug,
        )
        J = result.jac
        JTJ = J.T @ J
        if np.linalg.matrix_rank(JTJ) == 2:
            sigma = np.std(result.fun)  # estimate from residuals
            cov = sigma ** 2 * np.linalg.inv(JTJ)
        else:
            cov = np.full((2, 2), np.nan)  # singular
        result.x = np.append(result.x, fixed_z)
        #resape to have 3x3 with zeros for new elements
        new_cov = np.full((3,3), 0.)
        new_cov[0:2,0:2] = cov
        return tuple(result.x), new_cov

    def calculate_3D_NLS(self, usefull_ranges, initial_guess=None):
        """
        Estimate 3D tag position using nonlinear least squares multilateration.

        Parameters:
            anchors: list of (x, y, z) coordinates of anchors
            distances: list of measured distances to the tag (same order as anchors)
            initial_guess: optional initial estimate (x, y, z)
            bounds: ((x_min, y_min, z_min), (x_max, y_max, z_max)) — optional
                     to constrain the search (useful for known environment limits)

        Returns:
            (x, y, z): optimized estimated position of the tag
        """
        # bounds = [(-10, -15, -1), (30, 15, 3)]
        bounds = None
        anchors = []
        distances = []
        for i, row in enumerate(usefull_ranges):
            xid = int(row[0])
            anchors.append(self.odom_data[(self.odom_data['id'] == xid) & (self.odom_data['Anchor'] == True)][
                ['px', 'py', 'pz']].iloc[0].to_numpy())
            distances.append(row[2])
        anchors = np.array(anchors)
        distances = np.array(distances)

        # If no initial guess, use centroid of anchors
        if np.any(np.isnan(initial_guess)):
            initial_guess = np.mean(anchors, axis=0)

        if bounds is not None:
            lower, upper = np.array(bounds[0]), np.array(bounds[1])
            initial_guess = np.clip(initial_guess, lower, upper)

        def residuals(pos):
            # Difference between measured and computed distances
            return np.linalg.norm(anchors - pos, axis=1) - distances
        # Solve nonlinear least squares
        result = least_squares(
            residuals,
            x0=initial_guess,
            # bounds=bounds,
            method='trf',  # Trust Region Reflective algorithm (good with bounds)
            loss='soft_l1',  # robust to outliers
            f_scale=0.5,
            verbose= self.debug,
        )
        J = result.jac
        JTJ = J.T @ J
        if np.linalg.matrix_rank(JTJ) == 3:
            sigma = np.std(result.fun)  # estimate from residuals
            cov = sigma ** 2 * np.linalg.inv(JTJ)
        else:
            cov = np.full((3, 3), np.nan)  # singular
        return tuple(result.x), cov

    def calculate_error_on_range(self, usefull_ranges, pos):
        error = []
        for r in usefull_ranges:
            if r[2] is not np.nan:
                anchor_pos = self.odom_data[(self.odom_data['id'] == r[0]) & (self.odom_data['Anchor'] == True)][
                    ['px', 'py', 'pz']].iloc[0].to_numpy()
                dist = np.linalg.norm(pos - anchor_pos)
                error.append(np.sqrt((dist-r[2])**2))
        return error

    def calculateSL1_ABMatrix(self, usefull_ranges):
        # anchors_ids = self.odom_data[self.odom_data['Anchor'] == True]['id'].unique()
        A = np.empty((0, 3))
        B = np.empty((0,))
        for i in range(len(usefull_ranges)):
            id = usefull_ranges[i][0]
            pose_1 = self.odom_data[(self.odom_data['id'] == id) & (self.odom_data['Anchor'] == True)][
                ['px', 'py', 'pz']].iloc[0].to_numpy()
            r_pose_1 = np.square(usefull_ranges[i][2] - np.linalg.norm(pose_1))
            for j in range(len(usefull_ranges) - 1 - i):
                id2 = usefull_ranges[len(usefull_ranges) - 1 - j][0]
                pose_2 = self.odom_data[(self.odom_data['id'] == id2) & (self.odom_data['Anchor'] == True)][
                    ['px', 'py', 'pz']].iloc[0].to_numpy()
                Arow = 2 * (pose_1 - pose_2)
                A = np.append(A, [Arow], axis=0)
                r_pose_2 = np.square(usefull_ranges[len(usefull_ranges) - 1 - j][2] - np.linalg.norm(pose_2) )
                diff = r_pose_1 - r_pose_2
                B = np.append(B, [diff], axis=0)
        AtA = np.matmul(np.transpose(A), A)
        AtB = np.matmul(np.transpose(A), B)
        X = np.linalg.solve(AtA, AtB)
        return X

    def smoothingfilter(self, window_size):
        for id in self.tag_gts['id'].unique():
            tag_data = self.tag_gts[self.tag_gts['id'] == id]
            new_pos = []
            for row in tag_data.itertuples():
                t = row.time
                window = tag_data[(tag_data['time'] >= t - window_size*1e3) & (tag_data['time'] <= t + window_size*1e3)]
                pos = window[['px', 'py', 'pz']].mean()
                new_pos.append([t, id, pos['px'], pos['py'], pos['pz'], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            self.tag_gts = self.tag_gts[self.tag_gts['id'] != id]
            self.tag_gts = pd.concat([self.tag_gts, pd.DataFrame(new_pos, columns=self.tag_gts_columns)], ignore_index=True)

    #################################################################################
    ####  VIO
    #################################################################################

    def set_vio_transformation(self,t, q,dt, id):
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
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
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
            qv = np.concatenate([np.zeros((v.shape[0], 1)), v], axis=1)  # (N,4)
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
        vio_pos = self.vio_data[self.vio_data['id'] == id][["T_imu_wrt_vio_x(m)","T_imu_wrt_vio_y(m)","T_imu_wrt_vio_z(m)"]].to_numpy()
        vio_pos_rot = quat_rotate(np.array(q), vio_pos)
        vio_pos_fit = vio_pos_rot + np.array(t)
        vio_time = self.vio_data[self.vio_data['id'] == id]['timestamp(ns)'].to_numpy() + dt*1e9
        self.vio_data.loc[self.vio_data['id'] == id, 'timestamp(ns)'] = vio_time
        # replace the pos in self.vio_data for id id
        self.vio_data.loc[self.vio_data['id'] == id, ["T_imu_wrt_vio_x(m)","T_imu_wrt_vio_y(m)","T_imu_wrt_vio_z(m)"]] = vio_pos_fit

    ###############################################################################
    ####  ARP
    ###############################################################################
    def get_vio_transform(self, id):
        vio_data = self.vio_data[self.vio_data['id'] == id]
        time_data = vio_data['timestamp(ns)'].to_numpy()

        velocity_data = vio_data[["vel_imu_wrt_vio_x(m/s)", "vel_imu_wrt_vio_y(m/s)", "vel_imu_wrt_vio_z(m/s)"]].to_numpy()
        velocity_data = vio_data[["gravity_vector_x(m/s2)", "gravity_vector_y(m/s2)", "gravity_vector_z(m/s2)"]].to_numpy()
        gyro_data = vio_data[["angular_vel_x(rad/s)", "angular_vel_y(rad/s)", "angular_vel_z(rad/s)"]].to_numpy()

        return velocity_data, gyro_data, time_data

    def get_all_imu_data(self):
        for id in self.odom_data['id'].unique():
            self.get_imu_data(id)

    def get_imu_data(self, id):
        odom_data = self.odom_data[self.odom_data['id'] == id].sort_values(by='time')

        print(id)

    def get_orientation_from_gt(self):
        for id in self.tag_gts['id'].unique():
            pass



    def stream_odom_from_gt(self, id, history=0.1, start_time = None, end_time =None, imu_bool = False):
        def row_cov_matrix(row):
            """Return 3x3 covariance matrix from a DataFrame row with cov_* columns."""
            return np.array([
                [row['cov_xx'], row['cov_xy'], row['cov_xz']],
                [row['cov_xy'], row['cov_yy'], row['cov_yz']],
                [row['cov_xz'], row['cov_yz'], row['cov_zz']]
            ])
        previous_imu_index = 0
        previous_theta = np.nan
        previous_theta_var = np.nan
        odom_id_data = self.odom_data[self.odom_data['id'] == id].sort_values(by='time', ignore_index=True)

        for row, window in self.stream_gt_data_id(id, history=history, start_time=start_time, end_time=end_time):
            state_row = [1] * 8  # p, theta, v, w, vx
            cov_matrix = np.ones((len(state_row), len(state_row)))
            if imu_bool:
                state_row = [1] * 14 # p, theta, v, w , vx, a , gyr
                cov_matrix = np.ones((len(state_row), len(state_row)))
            cov_matrix = cov_matrix*np.nan
            state_row = np.array(state_row, dtype=float)*np.nan
            first = window.iloc[0]
            last = window.iloc[-1]
            t= (last["time"] / 1e3) - (history / 2)
            n = len(window)


            Sigma_first = row_cov_matrix(first)
            Sigma_last = row_cov_matrix(last)

            # p and its covariance
            p = np.mean(window[['px', 'py', 'pz']].to_numpy(), axis=0)
            state_row[:3] = p.tolist()
            sum_cov = np.zeros((3, 3))
            for i_row, row_i in window.iterrows():
                sum_cov += row_cov_matrix(row_i)
            p_cov =  sum_cov / n**2
            cov_matrix[0:3, 0:3] = p_cov

            # v and its covariance
            dx = np.array([last['px'] - first['px'], last['py'] - first['py'], last['pz'] - first['pz']])
            cov_dx = Sigma_last + Sigma_first
            cov_p_dx = (1.0 / n) * (Sigma_last - Sigma_first)

            v = dx / history
            state_row[4:7] = v.tolist()
            v_cov = (1/(history**2)) * cov_dx
            cov_matrix [4:7, 4:7] = v_cov
            cov_p_v = 1/history * cov_p_dx
            cov_matrix[0:3, 4:7] = cov_p_v
            cov_matrix[4:7, 0:3] = cov_p_v

            # TODO: IMPORTANT assumption that the drone and ARMS only moved forward, which was true in the experiments. But is not general.
            r2 = dx[0] ** 2 + dx[1] ** 2
            if r2 < 1 or (np.isnan(r2)):
                theta = np.nan
                state_row[3] = theta
                # degenerate — return large variance or handle specially
                var_theta = np.nan
                cov_theta_p = np.ones((1, 3))*np.nan
                cov_theta_v = np.ones((1, 3))*np.nan
            else:
                theta= np.arctan2(dx[1], dx[0])
                state_row[3] = theta
                J_theta = np.array([- dx[1] / r2, dx[0] / r2])  # size (2,)
                # extract 2x2 covariance of dx (x,y)
                cov_dx_xy = cov_dx[:2, :2]
                var_theta = float(J_theta @ cov_dx_xy @ J_theta.T)  # scalar

                # covariance between theta and p:
                # Cov([dx_x,dx_y], p) is first two rows of cov_p_dx transposed?
                # cov_p_dx is Cov(p,dx) (3x3). We need Cov([dx_x,dx_y], p) which is the transpose of Cov(p,[dx_x,dx_y])
                # So Cov([dx_x,dx_y], p) = cov_p_dx[:2, :].T
                cov_dxxy_p = cov_p_dx[:2, :].T  # shape (3,2) ? careful: cov_p_dx is (3,3) = Cov(p,dx)
                # We want Cov(theta, p) = J_theta * Cov([dx_x,dx_y], p)
                # J_theta shape (2,), Cov([dx_x,dx_y], p) shape (2,3) -> result (3,) after multiplication
                cov_theta_p = (J_theta @ cov_p_dx[:2, :])  # shape (3,)
                # Cov(theta, v) similarly: Cov([dx_x,dx_y], v) = history * Cov([dx_x,dx_y], dx) = history * cov_dx[:2,:]
                cov_theta_v = (J_theta @ (history * cov_dx[:2, :]))  # shape (3,)

            cov_matrix[3, 3] = var_theta
            cov_matrix[3, 0:3] = cov_theta_p
            cov_matrix[0:3, 3] = cov_theta_p
            cov_matrix[3, 4:7] = cov_theta_v
            cov_matrix[4:7, 3] = cov_theta_v

            if  not np.isnan(theta) and not np.isnan(previous_theta):
                a0, a1 = previous_theta, theta
                Dtheta = np.unwrap([a0, a1])[1] - np.unwrap([a0, a1])[0]
                previous_theta = theta
                previous_theta_var = var_theta
            else:
                Dtheta = np.nan
            w =  Dtheta / history
            state_row[7] = w

            var_prev = 0.0 if previous_theta_var is None else previous_theta_var
            # assume no covariance between prev and current if not tracked
            var_Dtheta = var_theta + var_prev
            var_wz = var_Dtheta / (history ** 2)

            cov_w = np.zeros((3, 3))
            cov_w[2, 2] = var_wz

            # cross-covariances between w and p/v come from scaling of theta covariances by 1/history
            cov_w_p = np.zeros((3, 3))
            cov_w_v = np.zeros((3, 3))
            if not np.isinf(var_theta):
                # cov(w_z, p_j) = cov(Dtheta/history, p_j) ~= cov(theta, p_j)/history  (prev treated as independent)
                cov_w_p[2, :] = cov_theta_p / history
                cov_w_v[2, :] = cov_theta_v / history
            cov_matrix[7, 0:3] = cov_w_p[2, :]
            cov_matrix[7, 4:7] = cov_w_v[2, :]
            cov_matrix[0:3, 7] = cov_w_p[2, :]
            cov_matrix[4:7, 7] = cov_w_v[2, :]
            cov_matrix[7,7] = var_wz

            if self.debug:
                print(f"Tag {id} at time {last['time']/1e3} s: Velocity = {v}, Angular Velocity = {w}")

            if imu_bool:
                # find clossest imu_data to t in self.imu_data for id:
                closest_idx = (odom_id_data['time'] - t*1e3).abs().idxmin()

                if previous_imu_index >= closest_idx:
                    print(f"Warning: No IMU data close to time {t} for id {id}")
                    state_row[8:14] = [np.nan]*6
                    cov_matrix[8:14, 8:14]  = np.ones((6,6))*np.nan
                else:
                    imu_rows = odom_id_data.loc[previous_imu_index:closest_idx]
                    a = np.mean(imu_rows[['ax', 'ay', 'az']].to_numpy(), axis=0)
                    g = np.mean(imu_rows[['gx', 'gy', 'gz']].to_numpy(), axis=0)
                    state_row[8:11] = a.tolist()
                    state_row[11:14] = g.tolist()
                    cov_matrix[8:14, 8:14] = np.eye(6)*len(imu_rows) # accelerometer noise
                    try:
                        previous_imu_index = closest_idx+1
                    except TypeError:
                        previous_imu_index = closest_idx

            yield t,state_row, cov_matrix


    def get_odom_from_gt(self, id , history=0.1, imu_bool = False, max_row = None):
        states = []
        cov_mats = []
        ts = []
        dt = history
        i = 0
        for t, state, cov_mat  in  self.stream_odom_from_gt(id, history=history, imu_bool=imu_bool):
            states.append(state)
            cov_mats.append(cov_mat)
            ts.append(t)

            if max_row is not None:
                i += 1
                if i >= max_row:
                    break

        return np.array(ts), np.array(states), np.array(cov_mats)



    #################################################################################
    ####  Streaming
    #################################################################################
    def stream_gt_data_id(self, id, history = 1, start_time = None, end_time = None):
        tag_data = self.tag_gts[self.tag_gts['id'] == id].sort_values(by='time')
        if start_time is not None:
            tag_data = tag_data[tag_data['time'] >= start_time*1e3]
        if end_time is not None:
            tag_data = tag_data[tag_data['time'] <= end_time*1e3]
        for i, row in tag_data.iterrows():
            t = row['time']
            window = tag_data[(tag_data['time'] >= t - history*1e3) & (tag_data['time'] <= t)]
            yield row, window

    def stream_data_id(self, id, history = 1):
        gt_data = self.tag_gts[self.tag_gts['id'] == id].sort_values(by='time')
        for i, row in gt_data.iterrows():
            t = row['time']
            window_gt = gt_data[(gt_data['time'] >= t - history*1e3) & (gt_data['time'] <= t)]
            window_imu = self.odom_data[(self.odom_data['id'] == id) & (self.odom_data['time'] >= t - history*1e3) & (self.odom_data['time'] <= t)]
            yield t, window_gt, window_imu

    def stream_exp(self, freq = None,  history=1):
        gt_data = self.tag_gts.sort_values(by='time')
        odom_data = self.odom_data.sort_values(by='time')
        t = 0
        for i, row in gt_data.iterrows():
            if freq is not None:
                if row["time"]  <= t + 1e3/freq:
                    continue
            t = row['time']
            window_gt = gt_data[(gt_data['time'] >= t - history*1e3) & (gt_data['time'] <= t)]
            window_odom = odom_data[(odom_data['time'] >= t - history*1e3) & (odom_data['time'] <= t)]
            yield t, window_gt, window_odom

    def stream_range_data(self, id, start_time = None, end_time = None, range_data = None):
        if range_data is None:
            range_data = self.range_data
        range_data = range_data[(range_data['id'] == id) | (range_data['rid'] == id)].sort_values(by='time')
        if start_time is not None:
            range_data = range_data[range_data['time'] >= start_time *1e3]
        if end_time is not None:
            range_data = range_data[range_data['time']  <= end_time*1e3]
        for i, row in range_data.iterrows():
            t = row['time']
            sid = row['id']
            rid = row['rid']
            if sid == id:
                other_id = rid
            else:
                other_id = sid
            range = row['dist_m']
            yield t/1e3, other_id, range, row['fp_rssi'], row['rx_rssi']

    def stream_exp_id(self, id, history, start_time = None, end_time = None):
        range_data = self.filter_all_ranges(id, time_horizon=0.1, max_std=0.1)
        range_streamer = self.stream_range_data(id, start_time=start_time, end_time=end_time, range_data=range_data)
        nan_distances = [np.nan] * int(np.max(range_data.id.unique()+1))
        t_r = 0
        for t, state_row, cov_mat in self.stream_odom_from_gt(id, history=history, start_time=start_time, end_time=end_time, imu_bool=True):
            distances = nan_distances.copy()
            while t_r < t:
                t_r, other_id, range, fp_rssi, rx_rssi = next(range_streamer)
                distances[int(other_id)] = range
            data ={ "time": t*1e3, "ranges": distances,
                    "state_row": state_row, "cov_mat": cov_mat}
            yield data





    #################################################################################
    ####  PLOTTING
    #################################################################################
    def plot_ranges(self, separate_plots = True):
        unique_ids = self.range_data['id'].unique()
        max_id = max(unique_ids)+1
        if not separate_plots:
            axs = plt.subplots(max_id, max_id, figsize=(15, 15), sharex=True, sharey=True)[1]
        for id in range(25):
            for rid in range(id+1,25):
                if id in unique_ids and rid in unique_ids:
                    if separate_plots:
                        self.plot_range(id, rid)
                    else:
                        self.plot_range(id, rid, ax=axs[id, rid])

    def plot_range(self, id, rid, ax = None, plot_rssi = False):
        if ax is None:
            ax = plt.figure().gca()
        data = self.range_data[(self.range_data['id'].isin([rid, id]))
                               & (self.range_data['rid'].isin([id, rid]))]
        if data.empty:
            print(f"No range data between {id} and {rid}")
            return
        ax.plot(data['time']*1e-3, data['dist_m'], ".", label=f"Range {id} to {rid}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title(f"Range to anchor {id}.")
        if plot_rssi:
            ax2 = ax.twinx()
            ax2.plot(data['time'], data['fp_rssi'], "r.", label=f"FP RSSI {id} to {rid}")
            ax2.plot(data['time'], data['rx_rssi'], "g.", label=f"RX RSSI {id} to {rid}")
            ax2.set_ylabel("RSSI (dBm)")
            ax2.legend(loc='upper right')
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Distance (m)")
        # ax.legend()

    def plot_filtered_range(self, id, rid, time_horizon=1, max_std=0.5):
        try:
            range_data, original, stds = self.filter_ranges(id, rid, time_horizon, max_std)
            plt.figure()
            plt.plot(stds)
            plt.figure()
            plt.plot(original['time'], original['dist_m'], 'r.')
            plt.plot(range_data['time'], range_data['dist_m'], 'b.')
            plt.twinx()
            plt.plot(original['time'], original["rx_rssi"], 'g-')
            plt.plot(original['time'], original["fp_rssi"], 'm-')
            plt.title(f"Filtered ranges between {id} and {rid}")
        except Exception as e:
            print(f"Error plotting filtered range between {id} and {rid}: {e}")

    def plot_filtered_ranges(self, time_horizon=1, max_std=1.0):
        unique_ids = self.range_data['id'].unique()
        cnt= 0
        for id in range(25):
            if id in unique_ids:
                if self.odom_data[self.odom_data['id'] == id]['Anchor'].any():
                    continue
                for rid in range(0,25):
                    if id is not rid and id in unique_ids and rid in unique_ids:
                        cnt += 1
                        self.plot_filtered_range(id, rid, time_horizon, max_std)
                        if cnt >= 10:
                            cnt = 0
                            plt.show()

    def plot_3D(self, name_dict= None):
        anchor_pose = self.odom_data[self.odom_data['Anchor'] == True][
                ['id', 'px', 'py', 'pz']].drop_duplicates().set_index('id').to_numpy()
        ax_3d = plt.figure().add_subplot(projection='3d')
        ax_3d.scatter(anchor_pose[:,0], anchor_pose[:,1], anchor_pose[:,2], c='k', label='Anchors')
        for id in self.tag_gts['id'].unique():
            tag_data = self.tag_gts[self.tag_gts['id'] == id]
            label = f'Tag {id} GT'
            if name_dict is not None and id in name_dict:
                label = f'{label} - {name_dict[id]}'
            ax_3d.plot(tag_data['px'], tag_data['py'], tag_data['pz'], label=label)
        try:
            for id in self.vio_data['id'].unique():
                vio_tag_data = self.vio_data[self.vio_data['id'] == id]
                ax_3d.plot(vio_tag_data['T_imu_wrt_vio_x(m)'], vio_tag_data['T_imu_wrt_vio_y(m)'], vio_tag_data['T_imu_wrt_vio_z(m)'], label=f'Tag {id} VIO', color='g')
        except Exception as e:
            print(f"Error plotting VIO data: {e}")

    def plot_gt_cov(self,  id):
        fig, ax = plt.subplots(3,2, figsize=(8,6), sharex=True)
        tag_data = self.tag_gts[self.tag_gts['id'] == id]
        # for i in range(3):
        ax[0,0].plot(tag_data['time'], tag_data['px'])
        ax[1,0].plot(tag_data['time'], tag_data['py'])
        ax[2,0].plot(tag_data['time'], tag_data['pz'])
        ax[0,1].plot(tag_data['time'], tag_data['cov_xx'])
        ax[0,1].plot(tag_data['time'], tag_data['cov_xy'])
        ax[0,1].plot(tag_data['time'], tag_data['cov_xz'])
        ax[1,1].plot(tag_data['time'], tag_data['cov_yy'])
        ax[1,1].plot(tag_data['time'], tag_data['cov_xy'])
        ax[1,1].plot(tag_data['time'], tag_data['cov_yz'])
        ax[2,1].plot(tag_data['time'], tag_data['cov_zz'])
        ax[2,1].plot(tag_data['time'], tag_data['cov_xz'])
        ax[2,1].plot(tag_data['time'], tag_data['cov_yz'])

    def plot_vio_data(self, id):
        if self.vio_data is None or self.vio_data.empty:
            print("No VIO data loaded.")
            return
        vio_tag_data = self.vio_data[self.vio_data['id'] == id]
        if vio_tag_data.empty:
            print(f"No VIO data for tag {id}.")
            return
        fig = plt.figure()
        ax_3d = fig.add_subplot(projection='3d')
        ax_3d.plot(vio_tag_data['T_imu_wrt_vio_x(m)'], vio_tag_data['T_imu_wrt_vio_y(m)'], vio_tag_data['T_imu_wrt_vio_z(m)'], label=f'Tag {id} VIO', color='g')

        plt.show()

    def plot_imu_data(self, id):
        fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        for i in range(3):
            imu_tag_data = self.odom_data[self.odom_data['id'] == id]
            if imu_tag_data.empty:
                print(f"No IMU data for tag {id}.")
                return
            axs[i, 0].plot(imu_tag_data['time'], imu_tag_data[['ax', 'ay', 'az']].to_numpy()[:, i], label=f'acc {["X","Y","Z"][i]}')
            axs[i, 0].set_ylabel(f'acc {["X","Y","Z"][i]} (uT)')
            axs[i, 0].legend()
            axs[i, 1].plot(imu_tag_data['time'], imu_tag_data[['gx', 'gy', 'gz']].to_numpy()[:, i], label=f'Gyro {["X","Y","Z"][i]}')
            axs[i, 1].set_ylabel(f'Gyro {["X","Y","Z"][i]} (rad/s)')
            axs[i, 1].legend()

    def stream_3D_plot_data(self, freq = 1, history=1, name_dict= None, color_dict= None, plot_bool = True, save_folder = None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        tags_list = self.tag_gts['id'].unique()
        i = 0

        for t, window_gt, window_odom in self.stream_exp(freq, history):
            ax.clear()
            fig.suptitle(f"Time: {t/1e3:.2f} s")
            anchor_pose = self.odom_data[self.odom_data['Anchor'] == True][
                ['id', 'px', 'py', 'pz']].drop_duplicates().set_index('id').to_numpy()
            ax.scatter(anchor_pose[:,0], anchor_pose[:,1], anchor_pose[:,2], c='k',marker="x", label='Anchors')
            for id in tags_list:
                if id in window_gt['id'].unique():
                    tag_data = window_gt[window_gt['id'] == id]
                    if color_dict is not None:
                        ax.plot(tag_data['px'], tag_data['py'], tag_data['pz'], label="Ground truth", color=color_dict[id])
                    else:
                        ax.plot(tag_data['px'], tag_data['py'], tag_data['pz'], label="Ground truth")

            ax.legend(loc='upper right', ncol=2)
            ax.view_init(elev=60, azim=180)
            ax.set_xlim(-10,35)
            ax.set_ylim(-15, 15)
            ax.set_zlim(-1, 5)
            # for id in window_odom['id'].unique():
            #     imu_tag_data = window_odom[window_odom['id'] == id]
            #     ax_3d.plot(imu_tag_data['px'], imu_tag_data['py'], imu_tag_data['pz'], label=f'Tag {id} Odom', linestyle='--')
            # ax_3d.set_title(f"Time: {t/1e3:.2f} s")
            # ax_3d.legend()
            if plot_bool:
                plt.pause(0.1)
                # plt.show()
            else:
                print(t)
            if save_folder is not None:
                plt.savefig(f"./{save_folder}/{i}.png")
            i+=1

    def stream_plot_data(self, freq = 1, history=1, name_dict= None, color_dict= None, plot_bool = True, save_folder = None):
        fig = plt.figure()
        tags_list = self.tag_gts['id'].unique()
        labels={}
        if name_dict is not None:
            labels = name_dict
        else:
            for id in tags_list:
                labels[id] = f'Tag {id} GT'

        i = 0

        for t, window_gt, window_odom in self.stream_exp(freq, history):

            plt.clf()
            plt.title(f"Time: {t/1e3:.2f} s")
            anchor_pose = self.odom_data[self.odom_data['Anchor'] == True][
                ['id', 'px', 'py', 'pz']].drop_duplicates().set_index('id').to_numpy()
            plt.scatter(anchor_pose[:,0], anchor_pose[:,1], c='k',marker="x", label='Anchors')
            for id in tags_list:
                if id in window_gt['id'].unique():
                    tag_data = window_gt[window_gt['id'] == id]
                    if color_dict is not None:
                        plt.plot(tag_data['px'], tag_data['py'], label=labels[id], color=color_dict[id])
                    else:
                        plt.plot(tag_data['px'], tag_data['py'], label=labels[id])

            plt.legend(loc='lower right', ncol=2)
            plt.xlim(-15, 35)
            plt.ylim(-15, 35)
            # for id in window_odom['id'].unique():
            #     imu_tag_data = window_odom[window_odom['id'] == id]
            #     ax_3d.plot(imu_tag_data['px'], imu_tag_data['py'], imu_tag_data['pz'], label=f'Tag {id} Odom', linestyle='--')
            # ax_3d.set_title(f"Time: {t/1e3:.2f} s")
            # ax_3d.legend()
            if plot_bool:
                plt.pause(0.1)
            else:
                print(t)
            if save_folder is not None:
                plt.savefig(f"./{save_folder}/{i}.png")
            i+=1
        # 2D plot of tag positions

    ###############################################################################
    ####  SAVING DATA
    ###############################################################################
    def save_gt(self, safe_folder):
        for id in self.tag_gts['id'].unique():
            tag_data = self.tag_gts[self.tag_gts['id'] == id]
            tag_data.to_csv(f"./{safe_folder}/tag_{id}_gt.csv")
            tag_data.to_pickle(f"./{safe_folder}/tag_{id}_gt.pkl")
            print(f"Saved tag {id} GT to ./{safe_folder}/tag_{id}_gt")

    def load_gt(self, safe_folder):
        self.tag_gts = pd.DataFrame(columns=self.tag_gts_columns)
        for file in os.listdir(safe_folder):
            if file.startswith("tag_") and file.endswith("_gt.csv"):
                id = int(file.split("_")[1])
                try:
                    tag_data = pd.read_csv(os.path.join(safe_folder, file))
                    self.tag_gts = pd.concat([self.tag_gts, tag_data], ignore_index=True)
                    print(f"Loaded tag {id} GT from {file}")
                except EmptyDataError:
                    print(f"No tag data loaded for tag {id}.")

    def load_odom_data(self, save_folder):
        self.odom_data = pd.DataFrame(columns=self.odom_columns)
        for file in os.listdir(save_folder):
            if file.startswith("tag_") and file.endswith("_imu.csv"):
                id = int(file.split("_")[1])
                try:
                    odom_data = pd.read_csv(os.path.join(save_folder, file))
                    self.odom_data = pd.concat([self.odom_data, odom_data], ignore_index=True)
                    print(f"Loaded device {id} odom from {file}")
                except EmptyDataError:
                    print(f"No odom data loaded for device {id}.")

    def load_vio_data(self, file_path, id):
        data = pd.read_csv(file_path)
        data['id'] = id
        try :
            self.vio_data = pd.concat([self.vio_data, data], ignore_index=True)
        except AttributeError:
            self.vio_data = data


    def load_imu_data(self, file_path, id, t_diff=0):
        # TODO: the imu_data should not exist, it is already in the odom_data.
        data = pd.read_csv(file_path)
        data['id'] = id
        data['time'] = data['timestamp(ns)']/1e6 + t_diff*1e3 # convert to ms and add time diff

        odom_data = pd.DataFrame(columns=self.odom_columns)
        odom_data['time'] = data['time']
        odom_data['id'] = data['id']
        odom_data['ax'] = data['AX(m/s2)']
        odom_data['ay'] = data['AY(m/s2)']
        odom_data['az'] = data['AZ(m/s2)']
        odom_data['gy'] = data['GX(rad/s)']
        odom_data['gx'] = data['GY(rad/s)']
        odom_data['gz'] = data['GZ(rad/s)']

        self.odom_data = pd.concat([self.odom_data[self.odom_data["id"]!=id], odom_data], ignore_index=True)


        # try :
        #     self.imu_data = pd.concat([self.imu_data, data], ignore_index=True)
        # except AttributeError:
        #     self.imu_data = data

    def save_range_data(self, file_path, rid, sid):
        range_data, original, stds = self.filter_ranges(rid, sid, 0.1, 0.3)
        range_data.to_csv(file_path+".csv", index=False)


@deprecation.deprecated(details="Moved to PD dataframes in Experiment class")
class Device():
    def __init__(self, id):
        self.id = id
        self.imu_data = []  # List to store IMU data over time
        self.ranges = {}  # Dictionary to store ranges to other devices {rid: (dist, fp_rssi, rx_rssi)}
        self.pos = np.array([0.0, 0.0, 0.0])

    def update_imu(self, t, mag, T, gyro, accel):
        self.imu_data.append([t, mag, T, gyro, accel])

    def update_range(self,t, rid, dist, fp_rssi, rx_rssi):
        if rid not in self.ranges:
            self.ranges[rid] = []
        self.ranges[rid].append([t, dist, fp_rssi, rx_rssi])
        self.arrange_ranges()

    def arrange_ranges(self):
        sorted_ranges = {}
        for rid in self.ranges:
            sorted_ranges[rid] = sorted(self.ranges[rid], key=lambda x: x[0])

    def plot_distances(self, id = None):
        pass

