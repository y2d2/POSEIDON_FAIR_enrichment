import deprecation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pos_Code.ESP_code.ESP_Class import ESP_wifi_module
from scipy.optimize import minimize


class Experiment():
    def __init__(self):
        self.odom_columns = ["time", "id", "mag_x", "mag_y", "mag_z", "T",
                            "gx", "gy", "gz", "ax", "ay", "az", "px", "py", "pz", "Anchor"]
        self.range_columns = ["time", "id", "rid", "dist_m", "fp_rssi", "rx_rssi"]
        self.odom_data = pd.DataFrame(columns=self.odom_columns)
        self.range_data = pd.DataFrame(columns=self.range_columns)

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
    #### ANCHOR SETUP
    #################################################################################

    def set_anchors(self, anchor_ids):
        #in self.odom_data set the pos of the anchors
        for anchor_id in anchor_ids:
            anchor_val = [anchor_ids[anchor_id][0], anchor_ids[anchor_id][1], anchor_ids[anchor_id][2], True]
            if int(anchor_id) in self.odom_data['id'].values:
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

    def calibrate_anchor_pos(self):
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
            error_matrix[std_dis_matrix > 0.05] = np.nan
            print(f"Current error: {np.nansum(np.abs(error_matrix))}")
            return np.nansum(np.abs(error_matrix))

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
        result = minimize(
            calculate_error,
            flattened_pos,
            args=(ordered_ids, mean_dis_matrix, std_dis_matrix),
            method='L-BFGS-B', options={'disp': True}
        )
        print(flattened_init_pos)
        print(np.abs(flattened_pos - result.x).reshape((-1,3)))

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
        ax.plot(data['time'], data['dist_m'], ".", label=f"Range {id} to {rid}")
        if plot_rssi:
            ax2 = ax.twinx()
            ax2.plot(data['time'], data['fp_rssi'], "r.", label=f"FP RSSI {id} to {rid}")
            ax2.plot(data['time'], data['rx_rssi'], "g.", label=f"RX RSSI {id} to {rid}")
            ax2.set_ylabel("RSSI (dBm)")
            ax2.legend(loc='upper right')
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Distance (m)")
        # ax.legend()





    def plot_3D(self):
        pass


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