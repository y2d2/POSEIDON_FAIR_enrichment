import unittest
import pandas as pd
import csv
import numpy as np
import Code.UtilityCode.Transformation_Matrix_Fucntions as TMF
from Code.UtilityCode.Measurement import create_experiment
import matplotlib.pyplot as plt

def create_experimental_data(data_frame:pd.DataFrame, sig_w, anchor_id, anchor_pos, start_time = None, end_time = None):
    if start_time is not None:
        data_frame = data_frame[data_frame['time'] >= start_time*1e3]
    if end_time is not None:
        data_frame = data_frame[data_frame['time'] <= end_time*1e3]

    experiment_data = {}
    experiment_data["name"] = "test"
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
        cov = np.ones((14,14))*np.nan
        for i in range(14):
            for j in range(14):
                cov[i,j] = row[f"cov_{i}_{j}"]
        if not np.isnan(heading):
            t_real = np.append(state[0:3], np.array(heading))
            T_real = TMF.transformation_matrix_from_4D_t(t_real)


        if not np.isnan(state[13]):
            dt_slam = np.array(state[4:7]+[state[-1]])*dt
            dt_slam[0] = np.linalg.norm(dt_slam[0:2])
            dt_slam[1] = 0
            DT_slam = DT_slam @ TMF.transformation_matrix_from_4D_t(dt_slam)

            q_slam = cov[4:8,4:8]
            q_slam[3,3] = sig_w**2 * cov[-1,-1]
            q_tot = np.sqrt(q_slam[0,0]**2 + q_slam[1,1]**2)
            q_slam[0,0] = q_tot
            q_slam[0:3,3] = [0,0,0]
            q_slam[3,0:3] = [0,0,0]
            q_slam = q_slam * dt**2
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

def get_experiment_data(file_path, anchor_id, start_time=None, end_time=None):
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
    data = load_experiment_data(file_path)
    exp = create_experimental_data(data, sig_w=0.0001, anchor_id=anchor_id,
                                   anchor_pos=call_positions[0],
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
        anchors_ids = {f"{i}": call_positions[i] for i in range(16)}
        data = load_experiment_data('./Data/full_exp.csv')
        exp = create_experimental_data(data, sig_w=0.08, anchor_id=0, anchor_pos=call_positions[0])

    def test_run_exp(self):
        sig_v = 0.08
        sig_w = 0.08
        sig_uwb = 0.25

        main_folder = "./Results/"
        results_folder = main_folder + "test"

        experiment_data = get_experiment_data('./Data/full_exp.csv', 0, end_time = None)

        methods = ["losupf|frequency=0|resample_factor=0.1|sigma_uwb_factor=1.0|multi_particles=0",
                   # "nodriftupf|frequency=10.0|resample_factor=0.1|sigma_uwb_factor=1.0",
                   # "algebraic|frequency=1.0|horizon=10",
                   # "algebraic|frequency=10.0|horizon=100",
                   # # "algebraic|frequency=10.0|horizon=1000",
                   # "QCQP|frequency=10.0|horizon=100",
                   # # "QCQP|frequency=10.0|horizon=1000",
                   # "NLS|frequency=1.0|horizon=10",
                   ]

        tas = create_experiment(results_folder, sig_v, sig_w, sig_uwb)
        tas.debug_bool = True
        tas.plot_bool = True
        tas.run_experiment(methods=methods, redo_bool=True, experiment_data=experiment_data)
        plt.show()

    def test_load_Aero_results(self):
        results_folder = "./Results/test/"
        # tas = create_experiment(results_folder, sig_v=0.08, sig_w=0.08, sig_uwb=0.25)
        # tas.load_results("./Results/test/")
if __name__ == '__main__':
    unittest.main()
