import os
import unittest
import pandas as pd
import csv
import numpy as np
from matplotlib.gridspec import GridSpec

import Code.UtilityCode.Transformation_Matrix_Fucntions as TMF
from Code.DataLoggers.ConnectedAgent_DataLogger import UPFConnectedAgentDataLogger
from Code.UtilityCode.Measurement import create_experiment
import matplotlib.pyplot as plt
import pickle as pkl

def create_experimental_data(data_frame:pd.DataFrame, sig_w, anchor_id, anchor_pos, start_time = None, end_time = None):
    if start_time is not None:
        data_frame = data_frame[data_frame['time'] >= start_time*1e3]
    if end_time is not None:
        data_frame = data_frame[data_frame['time'] <= end_time*1e3]

    experiment_data = {}
    experiment_data["name"] = "New_test"
    experiment_data["sample_freq"] = np.nan
    experiment_data["drones"] = {}
    heading = np.nan
    DT_anchor_list = []
    T_anchor_list = []
    Q_anchor_list = []
    uwb = []
    previous_t = None
    DT_slam = np.eye(4)
    Q_slam = np.zeros((4,4))
    DT_slam_list = []
    Q_slam_list = []
    T_real_list = []
    time_list = []
    dt_list = []
    for row_i,row in data_frame.iterrows():
        time = row['time']*1e-3
        if previous_t is None:
            dt = 0
            previous_t = time
        else:
            dt = time - previous_t
            previous_t = time
        ranges = [row[f"range_{i}"] for i in range(16) ]
        state = [row[f"state_{i}"] for i in range(14) ]
        if not np.isnan(state[3]):
            heading = state[3]
        # elif not np.isnan(state[13]):
        #     heading = heading - state[-1]*dt
        cov = np.ones((14,14))*np.nan
        for i in range(14):
            for j in range(14):
                cov[i,j] = row[f"cov_{i}_{j}"]
        if not np.isnan(heading):
            t_real = np.append(state[0:3], np.array(heading))
            T_real = TMF.transformation_matrix_from_4D_t(t_real)


        if not np.isnan(state[13]):
            dt_slam = np.array(state[4:7]+[-state[-1]])*dt
            dt_slam[0] = np.linalg.norm(dt_slam[0:2])
            dt_slam[1] = 0
            DT_slam = DT_slam @ TMF.transformation_matrix_from_4D_t(dt_slam)

            q_slam = cov[4:8,4:8]
            q_slam[3,3] = sig_w**2 * cov[-1,-1]
            q_tot = q_slam[0,0] + q_slam[1,1]
            q_slam[0,0] = q_tot
            q_slam[0:3,3] = [0,0,0]
            q_slam[3,0:3] = [0,0,0]
            q_slam = q_slam * dt**2
            # q_slam[:3,:3] = q_slam[:3,:3] *0.1
            R = TMF.get_rotation(DT_slam)
            T_rot = TMF.transformation_matrix_from_R(R, np.zeros(3))
            Q_slam = Q_slam + T_rot.T @ q_slam @ T_rot

        if not np.isnan(ranges[anchor_id]):
            DT_slam_list.append(DT_slam)
            T_real_list.append(T_real)
            Q_slam_list.append(Q_slam)
            Q_anchor_list.append(np.zeros((4,4)))
            T_anchor_list.append(TMF.transformation_matrix_from_4D_t(np.append(anchor_pos, 0)))
            DT_anchor_list.append(np.eye(4))
            uwb.append(ranges[anchor_id])
            DT_slam = np.eye(4)
            Q_slam = np.zeros((4, 4))
            time_list.append(time)
            print(f"{time}: UWB measurement: {ranges[anchor_id]}")


    while True:
        if np.all(Q_slam_list[1]==0):
            DT_slam_list.pop(0)
            T_real_list.pop(0)
            Q_slam_list.pop(0)
            DT_anchor_list.pop(0)
            Q_anchor_list.pop(0)
            T_anchor_list.pop(0)
            time_list.pop(0)
            uwb.pop(0)
        else:
            DT_slam_list.pop(0)
            Q_slam_list.pop(0)
            DT_anchor_list.pop(0)
            Q_anchor_list.pop(0)
            break

    experiment_data["drones"]["drone_0"] = {"DT_slam": DT_slam_list, "T_real": T_real_list, "Q_slam": Q_slam_list}
    experiment_data["drones"]["drone_1"] = {"DT_slam": DT_anchor_list, "T_real": T_anchor_list, "Q_slam": Q_anchor_list}
    experiment_data["uwb"] = uwb
    experiment_data["time"] = time_list
    # experiment_data["los_state"] = uwb_los
    # experiment_data["uwb_error"] = measurement.get_uwb_error()
    return experiment_data

def load_experiment_data(file_path):
    data = pd.read_csv(file_path)
    return data

def get_experiment_data(file_path, anchor_id, sig_w, start_time=None, end_time=None):
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

    call_positions = [
        [0.271, 3.582, 2.897],
        [0.212, 4.286, 0.190],
        [0.233, 1.079, 3.192],
        [0.193, 0.257, 0.468],
        [31.731, 3.102, 2.815],
        [4.857, 1.035, 2.761],
        [8.036, 4.377, 1.685],
        [8.082, -0.570, 2.953],
        [12.025, 1.182, 2.751],
        [12.134, 3.014, 2.870],
        [15.828, 4.291, 0.511],
        [16, 4.5 - 0.89, 3],# 11 not present in large data set.
        [15.947, -0.036, 0.148],
        [15.847, 1.097, 2.883],
        [7.956, -8.876, 2.105],
        [7.902, 13.200, 2.092],
    ]
    anchors_ids = {f"{i}": call_positions[i] for i in range(16)}
    data = load_experiment_data(file_path)
    exp = create_experimental_data(data, sig_w=sig_w, anchor_id=anchor_id,
                                   anchor_pos=call_positions[anchor_id],
                                   start_time=start_time, end_time=end_time)
    return exp

class MyTestCase(unittest.TestCase):
    def test_data_loading(self):
        data = load_experiment_data('./Data/full_exp.csv')

    def test_get_id_for_anchors(self):
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
        call_positions = [
            [0.271, 3.582, 2.897],
            [0.212, 4.286, 0.190],
            [0.233, 1.079, 3.192],
            [0.193, 0.257, 0.468],
            [31.731, 3.102, 2.815],
            [4.857, 1.035, 2.761],
            [8.036, 4.377, 1.685],
            [8.082, -0.570, 2.953],
            [12.025, 1.182, 2.751],
            [12.134, 3.014, 2.870],
            [15.828, 4.291, 0.511],
            [16, 4.5 - 0.89, 3],  # 11 not present in large data set.
            [15.947, -0.036, 0.148],
            [15.847, 1.097, 2.883],
            [7.956, -8.876, 2.105],
            [7.902, 13.200, 2.092],
        ]
        anchors_ids = {f"{i}": call_positions[i] for i in range(16)}
        data = load_experiment_data('./Data/full_exp.csv')
        exp = create_experimental_data(data, sig_w=0.08, anchor_id=0, anchor_pos=call_positions[0])

    def test_run_exp(self):
        sig_v = 0.08
        sig_w = 0.1
        sig_uwb = 0.2
        achor_ids = range(16)
        for anchor_id in achor_ids:
            if anchor_id ==11:
                continue
        # anchor_id = 6
            main_folder = "./Results/"
            results_folder = main_folder + "test"

            experiment_data = get_experiment_data('./Data/Aero_part_exp.csv',sig_w=sig_w, anchor_id=anchor_id,start_time=2660, end_time = 2700)
            experiment_data["name"] = f"Aero_test_exp_test5_{anchor_id}"
            methods = ["losupf|frequency=0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                       # "nodriftupf|frequency=0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                       # "algebraic|frequency=1.0|horizon=10",
                       # "algebraic|frequency=10.0|horizon=100",
                       # # "algebraic|frequency=10.0|horizon=1000",
                       # "QCQP|frequency=10.0|horizon=100",
                       # # "QCQP|frequency=10.0|horizon=1000",
                       # "NLS|frequency=1.0|horizon=10",
                       ]

            tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
            tas.debug_bool = True
            tas.save_bool=True
            tas.save_folder = results_folder
            tas.plot_bool = True
            tas.run_experiment(methods=methods, redo_bool=True, experiment_data=experiment_data)
        # plt.show()

    def test_plot_aero_GT(self):
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
        call_positions = [
            [0.271, 3.582, 2.897],
            [0.212, 4.286, 0.190],
            [0.233, 1.079, 3.192],
            [0.193, 0.257, 0.468],
            [31.731, 3.102, 2.815],
            [4.857, 1.035, 2.761],
            [8.036, 4.377, 1.685],
            [8.082, -0.570, 2.953],
            [12.025, 1.182, 2.751],
            [12.134, 3.014, 2.870],
            [15.828, 4.291, 0.511],
            [16, 4.5 - 0.89, 3],  # 11 not present in large data set.
            [15.947, -0.036, 0.148],
            [15.847, 1.097, 2.883],
            [7.956, -8.876, 2.105],
            [7.902, 13.200, 2.092],
        ]
        anchors_ids = {f"{i}": call_positions[i] for i in range(16)}
        data_frame = load_experiment_data('./Data/Aero_full_exp_h_01.csv')
        states = []
        cov_mats = []
        times = []
        max_row = None
        for i, row in data_frame.iterrows():
            time = row['time'] * 1e-3
            times.append(time)
            state = [row[f"state_{j}"] for j in range(14)]
            cov = np.ones((14, 14)) * np.nan
            for m in range(14):
                for n in range(14):
                    cov[m, n] = row[f"cov_{m}_{n}"]
            cov_mats.append(cov)
            states.append(state)

            if max_row is not None and i >= max_row:
                break
        states = np.array(states)
        cov_mats = np.array(cov_mats)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states[:, 0], states[:, 1], states[:,2],label='Ground Truth Trajectory', color="tab:blue")
        ax.scatter([call_positions[i][0] for i in range(len(call_positions))],
                    [call_positions[i][1] for i in range(len(call_positions))],
                    [call_positions[i][2] for i in range(len(call_positions))],
                    label="Anchor Positions", color="r", marker="x")
        ax.legend()
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        fig.suptitle(f"Greenhouse ground truth")

        fig, ax = plt.subplots(4, 3, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Greenhouse ground truth")
        ax[0, 0].plot(times, states[:, 0], label="X")
        ax[1, 0].plot(times, states[:, 1], label="Y")
        ax[2, 0].plot(times, states[:, 2], label="z")
        ax[0, 0].set_ylabel("X Position (m)")
        ax[1, 0].set_ylabel("Y Position (m)")
        ax[2, 0].set_ylabel("Y Position (m)")
        ax[3, 0].plot(times, states[:, 3], label="H")
        ax[3, 0].set_ylabel("Heading (rad)")

        ax[0, 1].plot(times, cov_mats[:, 0, 0], label="Cov XX")
        ax[0, 1].plot(times, cov_mats[:, 0, 1], label="Cov XY")
        ax[0, 1].plot(times, cov_mats[:, 0, 2], label="Cov Xz")
        ax[1, 1].plot(times, cov_mats[:, 1, 1], label="Cov YY")
        ax[1, 1].plot(times, cov_mats[:, 0, 1], label="Cov XY")
        ax[1, 1].plot(times, cov_mats[:, 1, 2], label="Cov YZ")
        ax[2, 1].plot(times, cov_mats[:, 2, 2], label="Cov ZZ")
        ax[2, 1].plot(times, cov_mats[:, 0, 2], label="Cov XZ")
        ax[2, 1].plot(times, cov_mats[:, 1, 2], label="Cov YZ")
        ax[0, 1].set_ylabel("X Covariance (m²)")
        ax[1, 1].set_ylabel("Y Covariance (m²)")
        ax[2, 1].set_ylabel("Z Covariance (m²)")
        ax[3, 1].plot(times, cov_mats[:, 3, 3], label="Cov H")
        ax[3, 1].set_ylabel("H Covariance ((rad)²)")

        ax[0, 2].plot(times, states[:, 8], label="Acc X")
        ax[1, 2].plot(times, states[:, 9], label="Acc Y")
        ax[2, 2].plot(times, states[:, 10], label="Acc Z")
        ax[3, 2].plot(times, states[:, 13], label="Gyr Z")
        ax[0, 2].set_ylabel("Acc X (m/s²)")
        ax[1, 2].set_ylabel("Acc Y (m/s²)")
        ax[2, 2].set_ylabel("Acc Y (m/s²)")
        ax[3, 2].set_ylabel("Gyr Z ((rad)/s)")

        for ax in ax.flatten():
            ax.legend()
            ax.set_xlabel("Time (s)")
        print(np.nanstd(states[:, 8]))
        print(np.nanstd(states[:, 9]))
        print(np.nanstd(states[:, -1]))


        plt.show()

    def test_load_Aero_results(self):
        i = -1
        for anchor_id in range(16):
            try:
                results_folder = f"./Results/test/Aero_test_exp_test2_{anchor_id}"
                for file in os.listdir(results_folder):
                    if file.endswith(".pkl"):
                        print(f"Loading file: {file}")
                        data_logger : UPFConnectedAgentDataLogger = pkl.load(open(results_folder + "/" + file, "rb"))
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        best_particle_log = data_logger.find_particle_log(data_logger.upf_connected_agent.best_particle).rpea_datalogger
                        best_particle_log.plot_ca_corrected_estimated_trajectory(ax, color = "gold", label="Active relative pose estimation")
                        data_logger.connected_agent.set_plotting_settings(color="k")
                        data_logger.connected_agent.plot_real_position(ax, annotation=f"Ground truth", alpha=1., i=i, history=None)
                        data_logger.host_agent.set_plotting_settings(color="darkgreen")
                        data_logger.host_agent.plot_real_position(ax, annotation=f"Anchor", i=i, history=None)
                        data_logger.connected_agent.plot_slam_position(ax, color="tab:blue", annotation="SLAM position",linestyle="-", alpha=1, i=i)
                        fig.suptitle(f"Experiment results for anchor ID {anchor_id}")
                        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        ax.view_init(elev=60, azim=120)

                        # data_logger.plot_self(title=f" Anchor ID: {anchor_id}")
            except Exception as e:
                print(f"Error loading results for anchor ID {anchor_id}: {e}")
        plt.show()


        # tas = create_experiment(results_folder, sig_v=0.08, sig_w=0.08, sig_uwb=0.25)
        # tas.load_results("./Results/test/")
if __name__ == '__main__':
    unittest.main()
