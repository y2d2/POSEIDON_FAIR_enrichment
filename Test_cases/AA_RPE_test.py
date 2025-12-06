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
    previous_w = np.nan
    w = np.nan
    w_cov = np.nan
    prev_w_cov  = np.nan
    for row_i,row in data_frame.iterrows():
        time = row['time']*1e-3
        if previous_t is None:
            dt = 0
            previous_t = time
        else:
            dt = time - previous_t
            previous_t = time
        ranges = [row[f"range_{i}"] for i in range(19) ]
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
        else:
            t_real = np.append(state[0:3], np.array(0))
        T_real = TMF.transformation_matrix_from_4D_t(t_real)

        if np.isnan(state[13]):
            w = previous_w
            w_cov = prev_w_cov
        else:
            w = -state[13]
            previous_w = w
            prev_w_cov = cov[13,13]
            w_cov=cov[13,13]

        if not np.isnan(w):
            dt_slam = np.array(state[4:7]+[w])*dt
            dt_slam[0] = np.linalg.norm(dt_slam[0:2])
            dt_slam[1] = 0
            DT_slam = DT_slam @ TMF.transformation_matrix_from_4D_t(dt_slam)

            q_slam = cov[4:8,4:8]
            q_slam[3,3] = sig_w**2 * w_cov
            q_tot = q_slam[0,0] + q_slam[1,1]
            q_slam[0,0] = q_tot
            q_slam[0:3,3] = [0,0,0]
            q_slam[3,0:3] = [0,0,0]
            q_slam = q_slam * dt**2
            q_slam[2,2] = 1e-10
            R = TMF.get_rotation(DT_slam)
            T_rot = TMF.transformation_matrix_from_R(R, np.zeros(3))
            Q_slam = Q_slam + T_rot.T @ q_slam @ T_rot

        if not np.isnan(ranges[anchor_id]) :
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
    call_positions = [
        [8.609, 12.037, 1.326],
        [0.000, 3.600, 1.869],
        [3.286, 23.873, 1.395],
        [13.692, 12.300, 1.424],
        [8.202, 7.523, 1.250],
        [13.586, 23.863, 1.400],
        [26.000, 18.304, 1.263],
        [25.498, 24.090, 1.354],
    ]
    anchors_ids = {f"{i}": call_positions[i] for i in range(len(call_positions))}
    data = load_experiment_data(file_path)
    exp = create_experimental_data(data, sig_w=sig_w, anchor_id=anchor_id,
                                   anchor_pos=call_positions[anchor_id],
                                   start_time=start_time, end_time=end_time)
    return exp

class MyTestCase(unittest.TestCase):
    def test_data_loading(self):
        data = load_experiment_data('./Data/full_exp.csv')

    # def test_get_id_for_anchors(self):
    #     call_positions = [
    #         [8.609, 12.037, 1.326],
    #         [0.000, 3.600, 1.869],
    #         [3.286, 23.873, 1.395],
    #         [13.692, 12.300, 1.424],
    #         [8.202, 7.523, 1.250],
    #         [13.586, 23.863, 1.400],
    #         [26.000, 18.304, 1.263],
    #         [25.498, 24.090, 1.354],
    #     ]
    #     anchors_ids = {f"{i}": call_positions[i] for i in range(len(call_positions))}
    #     data = load_experiment_data(f'./Data/full_exp_{tag_id}.csv')
    #     exp = create_experimental_data(data, sig_w=0.08, anchor_id=0, anchor_pos=call_positions[anchor_id])

    def test_run_exp(self):
        sig_v = 0.08
        sig_w = 0.01
        sig_uwb = 0.3

        name_dict= {19: 6344, 18:7076, 17 : 7075,
                     16 :6169 , 15: 6192,13:6179,
                    12: 6167, 11: 6168, 10: 6369,
                    9: 6164, 8: 6184}
        tag_id = 8
        achor_ids = range(8)
        for anchor_id in achor_ids:
            if anchor_id ==11:
                continue
        # anchor_id = 6
            main_folder = "./Results/"
            results_folder = main_folder + "test_AA"

            experiment_data = get_experiment_data(f'./AA_monday_exp/full_exp_{tag_id}.csv',sig_w=sig_w, anchor_id=anchor_id,
                                                  start_time=4298, end_time = 4360)
            experiment_data["name"] = f"AA_test_exp_{tag_id}_{anchor_id}"
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

    def test_plot_AA_GT(self):
        tag_id = 8
        call_positions = [
            [8.609, 12.037, 1.326],
            [0.000, 3.600, 1.869],
            [3.286, 23.873, 1.395],
            [13.692, 12.300, 1.424],
            [8.202, 7.523, 1.250],
            [13.586, 23.863, 1.400],
            [26.000, 18.304, 1.263],
            [25.498, 24.090, 1.354],
        ]
        name_dict = {19: 6344, 18: 7076, 17: 7075,
                     16: 6169, 15: 6192, 13: 6179,
                     12: 6167, 11: 6168, 10: 6369,
                     9: 6164, 8: 6184}

        anchors_ids = {f"{i}": call_positions[i] for i in range(len(call_positions))}
        data_frame = load_experiment_data(f'./AA_monday_exp/full_exp_{tag_id}.csv')
        states = []
        cov_mats = [ ]
        times = []
        max_row = None
        for i, row in data_frame.iterrows():
            time = row['time'] * 1e-3
            times.append(time)
            state = [row[f"state_{j}"] for j in range(14)]
            cov = np.ones((14,14))*np.nan
            for m in range(14):
                for n in range(14):
                    cov[m,n] = row[f"cov_{m}_{n}"]
            cov_mats.append(cov)
            states.append(state)

            if max_row is not None and i >= max_row:
                break
        states = np.array(states)
        cov_mats = np.array(cov_mats)
        fig = plt.figure()
        plt.plot(states[:,0], states[:,1], label='Ground Truth Trajectory', color="tab:blue")
        plt.scatter([call_positions[i][0] for i in range(len(call_positions))],
                        [call_positions[i][1] for i in range(len(call_positions))],
                        label="Anchor Positions", color="r", marker="x")
        plt.legend()
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        fig.suptitle(f"E-logistics ground truth for ARM {name_dict[tag_id]}")


        fig, ax = plt.subplots(3,3, figsize=(10,8), sharex=True)
        fig.suptitle(f"E-logistics ground truth for ARM {name_dict[tag_id]}")
        ax[0,0].plot(times, states[:,0], label="X")
        ax[1,0].plot(times, states[:,1], label="Y")
        ax[0,0].set_ylabel("X Position (m)")
        ax[1,0].set_ylabel("Y Position (m)")
        ax[2,0].plot(times, states[:,3], label="H")
        ax[2,0].set_ylabel("Heading (rad)")

        ax[0,1].plot(times, cov_mats[:,0,0], label="Cov XX")
        ax[0,1].plot(times, cov_mats[:,0,1], label="Cov XY")
        ax[1,1].plot(times, cov_mats[:,1,1], label="Cov YY")
        ax[1,1].plot(times, cov_mats[:,0,1], label="Cov XY")
        ax[0,1].set_ylabel("X Covariance (m²)")
        ax[1,1].set_ylabel("Y Covariance (m²)")
        ax[2,1].plot(times, cov_mats[:,3,3], label="Cov H")
        ax[2,1].set_ylabel("H Covariance ((rad)²)")

        ax[0,2].plot(times, states[:,8], label="Acc X")
        ax[1,2].plot(times, states[:,9], label="Acc Y")
        ax[2,2].plot(times, states[:,13], label="Gyr Z")
        ax[0,2].set_ylabel("Acc X (m/s²)")
        ax[1,2].set_ylabel("Acc Y (m/s²)")
        ax[2,2].set_ylabel("Gyr Z ((rad)/s)")

        for ax in ax.flatten():
            ax.legend()
            ax.set_xlabel("Time (s)")
        print(np.nanstd(states[:,8]))
        print(np.nanstd(states[:,9]))
        print(np.nanstd(states[:,-1]))

        # fig = plt.figure(figsize=(15, 12))  # , layout="constrained")
        # fig.suptitle("Greenhouse ground truth and uncertainty")
        # ax = []
        # gs = GridSpec(4, 5, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1,1])
        # ax_3d = fig.add_subplot(gs[:4, :3], projection="3d")
        # ax_3d.plot(states[], py, pz, label="Ground truth trajectory", color="tab:blue")
        # ax_3d.scatter([call_positions[i][0] for i in range(len(call_positions))],
        #                 [call_positions[i][1] for i in range(len(call_positions))],
        #                 [call_positions[i][2] for i in range(len(call_positions))],
        #                 label="Anchor Positions", color="r", marker="x")
        # ax_3d.legend()
        # # ax = fig.add_subplot(gs[3, :4])
        # ax = [fig.add_subplot(gs[i, 3]) for i in range(4)]
        # ax[0].plot(times, px, label="X Position")
        # ax[1].plot(times, py, label="Y Position")
        # ax[2].plot(times, pz, label="Z Position")
        # ax[3].plot(times, h, label="Heading")
        # for a in ax:
        #     a.legend()
        #     a.grid()
        #     a.set_xlabel("Time (s)")
        #     a.set_ylabel("Position (m)")
        # ax[3].set_ylabel("Heading (rad)")
        #
        # ax = [fig.add_subplot(gs[i, 4]) for i in range(4)]
        # ax[0].plot(times, c_xx, label="Cov XX")
        # ax[0].plot(times, c_xy, label="Cov XY")
        # ax[0].plot(times, c_xz, label="Cov XZ")
        # ax[1].plot(times, c_yy, label="Cov YY")
        # ax[1].plot(times, c_yz, label="Cov YZ")
        # ax[1].plot(times, c_xy, label="Cov ZZ")
        # ax[2].plot(times, c_zz, label="Cov ZZ")
        # ax[2].plot(times,c_xz, label="Cov ZX")
        # ax[2].plot(times, c_yz, label="Cov ZY")
        # ax[3].plot(times,c_h,label="Cov H")
        # for a in ax:
        #     a.legend()
        #     a.grid()
        #     a.set_xlabel("Time (s)")
        #     a.set_ylabel("Covariance (m^2)")
        # ax[3].set_ylabel("Covariance (rad^2)")


        plt.show()

    def test_plot_AA_states(self):
        pass

    def test_load_Aero_results(self):
        tag_id = 8
        name_dict= {19: 6344, 18:7076, 17 : 7075,
                     16 :6169 , 15: 6192,13:6179,
                    12: 6167, 11: 6168, 10: 6369,
                    9: 6164, 8: 6184}
        for anchor_id in range(8):
            try:
                results_folder = f"./Results/test_AA/AA_test_exp_{tag_id}_{anchor_id}"
                for file in os.listdir(results_folder):
                    if file.endswith(".pkl"):
                        print(f"Loading file: {file}")
                        data_logger : UPFConnectedAgentDataLogger = pkl.load(open(results_folder + "/" + file, "rb"))
                        # fig = plt.figure()
                        # ax = fig.add_subplot(111, projection='3d')
                        # best_particle_log = data_logger.find_particle_log(data_logger.upf_connected_agent.best_particle).rpea_datalogger
                        # best_particle_log.plot_ca_corrected_estimated_trajectory(ax, color = "gold", label="Active relative pose estimation")
                        # data_logger.connected_agent.set_plotting_settings(color="k")
                        # data_logger.connected_agent.plot_real_position(ax, annotation=f"Ground truth ARM {name_dict[tag_id]}", alpha=1., i=-1, history=None)
                        # data_logger.host_agent.set_plotting_settings(color="darkgreen")
                        # data_logger.host_agent.plot_real_position(ax, annotation=f"Anchor {anchor_id}", i=-1, history=None)
                        # data_logger.connected_agent.plot_slam_position(ax, color="tab:blue", annotation="SLAM position",linestyle="-", alpha=1, i=-1)
                        # fig.suptitle(f"AA experiment results for tag ID {tag_id} (ARM {name_dict[tag_id]}) and anchor ID {anchor_id}")
                        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        # ax.view_init(elev=90, azim=-90)
                        data_logger.plot_self(title=f"AA experiment results for tag ID {tag_id} (ARM {name_dict[tag_id]}) and anchor ID {anchor_id}")
            except Exception as e:
                print(f"Error loading results for anchor ID {anchor_id}: {e}")
        plt.show()


        # tas = create_experiment(results_folder, sig_v=0.08, sig_w=0.08, sig_uwb=0.25)
        # tas.load_results("./Results/test/")
if __name__ == '__main__':
    unittest.main()
