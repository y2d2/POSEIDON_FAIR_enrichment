import unittest
import pos_Code.Experimental_Setup.Experimental_Setup as Experimental_Setup
import matplotlib.pyplot as plt
import numpy as np


def load_rough_aero_setup():
    rough_positions = [[0, 4.5 - 0.86, 3],
                       [0, 4.5 - 0.14, 0.45],
                       [0, 0.94, 3],
                       [0, 0.1, 0.43],
                       [32, 4.5 - 1.35, 3],
                       [8 - 3.3, 0.99, 2.9],
                       [8, 4.5 - 0.025, 1.65],
                       [8, -0.87, 3],
                       [8 + 4.14, 1, 0.97, 2.9],
                       [8 + 4.14, 4.5 - 1.3, 2.9],
                       [16, 4.5 - 0.025, 0.36],
                       [16, 4.5 - 0.89, 3],
                       [16, 0, 0.28],
                       [16, 1.02, 3],
                       [8, -9, 1.96],
                       [8, 4.5 + 9, 1.80],
                       ]
    anchors_ids = {f"{i}": rough_positions[i] for i in range(16)}
    exp_setup = Experimental_Setup.Experiment()
    folder_path = "../ESP_code/data/Long Experiment noon"
    exp_setup.load_data(folder_path)
    exp_setup.set_anchors(anchors_ids)
    return exp_setup


def load_cal_aero_setup(folder_path= "../ESP_code/data/Aerolytics_small_extract", debug=False):
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
    exp_setup = Experimental_Setup.Experiment(debug)
    # folder_path = "../ESP_code/data/small_extract"
    exp_setup.load_data(folder_path)
    exp_setup.set_anchors(anchors_ids)
    return exp_setup


def load_AA_rough_setup(debug=False):
    rough_positions = [[8, 12, 1.35],
                       [0, 4, 1.82],
                       [3, 23, 1.4],
                       [13, 12, 1.38],
                       [8, 7, 1.3],
                       [13, 23, 1.45],
                       [26, 17, 1.25],
                       [26, 22, 1.35],
                       ]
    bounds_list = []
    for pos in rough_positions:
        bounds_list.append((pos[0] - 0.1 * pos[0], pos[0] + 0.1 * pos[0]))
        bounds_list.append((pos[1] - 0.1 * pos[1], pos[1] + 0.1 * pos[1]))
        bounds_list.append((pos[2] - 0.05, pos[2] + 0.05))
    anchors_ids = {f"{i}": rough_positions[i] for i in range(len(rough_positions))}
    exp_setup = Experimental_Setup.Experiment(debug)
    folder_path = "../ESP_code/data/AA_small_extract"
    folder_path = "../ESP_code/data/AA_monday"
    exp_setup.load_data(folder_path)
    exp_setup.set_anchors(anchors_ids)
    return exp_setup, bounds_list


def load_AA_cal_setup(debug=False):
    cal_positions = [
        [8.609, 12.037, 1.326],
        [0.000, 3.600, 1.869],
        [3.286, 23.873, 1.395],
        [13.692, 12.300, 1.424],
        [8.202, 7.523, 1.250],
        [13.586, 23.863, 1.400],
        [26.000, 18.304, 1.263],
        [25.498, 24.090, 1.354],
    ]
    anchors_ids = {f"{i}": cal_positions[i] for i in range(len(cal_positions))}
    exp_setup = Experimental_Setup.Experiment(debug)
    folder_path = "../ESP_code/data/AA_monday"
    folder_path = "../ESP_code/data/AA_small_extract"
    exp_setup.load_data(folder_path)
    exp_setup.set_anchors(anchors_ids)
    return exp_setup


class MyTestCase(unittest.TestCase):

    ##################################################################################
    #### LOADING AND CHECKING DATA
    ##################################################################################

    def test_read_data(self):
        exp_setup = Experimental_Setup.Experiment(debug=True)
        folder_path = "../ESP_code/data/AA_friday_2"
        exp_setup.load_data(folder_path)
        # print(exp_setup.odom_data)

    def test_check_frequencies(self):
        exp_setup = Experimental_Setup.Experiment()
        folder_path = "../ESP_code/data/Long Experiment noon"
        exp_setup.load_data(folder_path)
        exp_setup.check_frequencies()

    ##################################################################################
    #### ANCHOR CALIBRATION
    ##################################################################################
    def test_set_anchor_ids(self):
        #Aerolytics setup:
        rough_positions = [[0, 4.5 - 0.86, 3],
                           [0, 4.5 - 0.14, 0.45],
                           [0, 0.94, 3],
                           [0, 0.1, 0.43],
                           [32, 4.5 - 1.35, 3],
                           [8 - 3.3, 0.99, 2.9],
                           [8, 4.5 - 0.025, 1.65],
                           [8, -0.87, 3],
                           [8 + 4.14, 1, 0.97, 2.9],
                           [8 + 4.14, 4.5 - 1.3, 2.9],
                           [16, 4.5 - 0.025, 0.36],
                           [16, 4.5 - 0.89, 3],
                           [16, 0, 0.28],
                           [16, 1.02, 3],
                           [8, -9, 1.96],
                           [8, 4.5 + 9, 1.80],
                           ]

        anchors_ids = {f"{i}": rough_positions[i] for i in range(16)}
        exp_setup = Experimental_Setup.Experiment()
        folder_path = "../ESP_code/data/udp_data_2025-09-18_11-13-41"
        exp_setup.load_data(folder_path)
        exp_setup.set_anchors(anchors_ids)
        print(exp_setup.odom_data[exp_setup.odom_data['Anchor'] == True]["id"].unique())

    def test_range_statistics(self):
        exp_setup, _ = load_AA_rough_setup()
        mean_matrix, std_matrix = exp_setup.get_anchor_ranges_statistics()
        print(mean_matrix)
        print(std_matrix)
        print(np.nansum(std_matrix<0.5))

    def test_anchor_cal(self):
        exp_setup = load_rough_aero_setup()
        exp_setup.calibrate_anchor_pos(bounds = 0.3, plot=True)

    def test_AA_anchor_cal(self):
        exp_setup, bounds_list = load_AA_rough_setup(debug=True)
        exp_setup.calibrate_anchor_pos(bounds = 0.3, plot=True, bounds_list=bounds_list)

    ##################################################################################
    ####  GROUND TRUTH TRAJECTORY
    ##################################################################################
    def test_range_filter(self):
        exp_setup = load_cal_aero_setup()
        range_data, original, stds = exp_setup.filter_ranges(0, 16, time_horizon = 1, max_std=1.0)
        plt.figure()
        plt.plot(stds)
        plt.figure()
        plt.plot(original['time'], original['dist_m'], 'r.')
        plt.plot(range_data['time'], range_data['dist_m'], 'b.')
        plt.show()

    def test_plot_filtered_ranges(self):
        # Filter seems not to be effective with the current data set.
        exp_setup = load_cal_aero_setup(debug = True)
        exp_setup.plot_filtered_ranges(time_horizon= 0.1, max_std=0.1)
        plt.show()

    def test_calculate_gt_trajectory(self):
        exp_setup = load_cal_aero_setup(debug = True)
        exp_setup.calculate_gt_trajectory_of_tag(16, time_horizon=1, max_std=0.8)
        print(exp_setup.tag_gts)

    def test_calculate_gt_trajectory_AA(self):
        exp_setup = load_AA_cal_setup(debug = True)
        exp_setup.calculate_gt_trajecotries(time_horizon=1, max_std=0.8)
        print(exp_setup.tag_gts)

    def test_filtering(self):
        exp_setup = load_cal_aero_setup()
        load_path = "../ESP_code/data/Aerolytics_small_extract_GT"
        exp_setup.load_gt(load_path)
        exp_setup.smoothingfilter(window_size=0.5)
        exp_setup.load_vio_data("../ESP_code/data/Long Experiment noon VIO/data.csv", 16)
        exp_setup.plot_3D()
        plt.show()


    ##################################################################################
    #### VIO
    ##################################################################################
    def test_set_init_transform(self):
        t = [12.63509959,  0.94427232,  1.59824491]
        q = [0.29832535, -0.73748568,  0.60547778,-0.02266103]
        gt_time_0 = 2627.448 #s
        vio_time_0 = 364.598 #s
        dt = 20.433 #s
        delta_t = gt_time_0 - vio_time_0 + dt
        exp_setup = load_cal_aero_setup()
        load_path = "../ESP_code/data/Aerolytics_small_extract_GT"
        exp_setup.load_gt(load_path)
        exp_setup.smoothingfilter(window_size=0.5)
        exp_setup.load_vio_data("../ESP_code/data/Long Experiment noon VIO/data.csv", 16)
        exp_setup.plot_3D()
        exp_setup.set_vio_transformation( t, q, delta_t, 16)
        exp_setup.plot_3D()
        plt.show()

    def test_get_vio_transform(self):
        t = [12.63509959,  0.94427232,  1.59824491]
        q = [0.29832535, - 0.73748568,  0.60547778,- 0.02266103]
        gt_time_0 = 2627.448 #s
        vio_time_0 = 364.598 #s
        dt = 20.433 #s
        delta_t = gt_time_0 - vio_time_0 + dt
        exp_setup = load_cal_aero_setup()
        load_path = "../ESP_code/data/Aerolytics_small_extract_GT"
        exp_setup.load_gt(load_path)
        exp_setup.smoothingfilter(window_size=0.5)
        exp_setup.load_vio_data("../ESP_code/data/Long Experiment noon VIO/data.csv", 16)
        exp_setup.set_vio_transformation(t, q, delta_t, 16)
        vio_t, vio_q, tim = exp_setup.get_vio_transform(16)
        _, ax  = plt.subplots(3,2)
        ax[0,0].plot(tim, vio_t[:,0])
        ax[1,0].plot(tim, vio_t[:,1])
        ax[2,0].plot(tim, vio_t[:,2])
        ax[0,1].plot(tim, vio_q[:,0])
        ax[1,1].plot(tim, vio_q[:,1])
        ax[2,1].plot(tim, vio_q[:,2])

        plt.show()

    ##################################################################################
    #### PLOTTING
    ##################################################################################
    def test_plot_range(self):
        exp_setup = load_rough_aero_setup()
        exp_setup.plot_range(6, 16, plot_rssi=False)
        plt.show()


    def test_plot_ranges(self):
        exp_setup= load_AA_cal_setup(debug=True)
        exp_setup.plot_ranges(separate_plots=True)
        plt.show()

    def test_3D_plot(self):
        exp_setup = load_cal_aero_setup(debug=True)
        exp_setup.calculate_gt_trajectory_of_tag(16, time_horizon=1, max_std=0.8)
        exp_setup.plot_3D()
        plt.show()

    def test_plot_vio(self):
        filepath = "../ESP_code/data/Long Experiment noon VIO/data.csv"
        exp_setup = load_cal_aero_setup()
        exp_setup.load_vio_data(filepath, 16)
        exp_setup.plot_vio_data(16)

    ###############################################################################
    ####  SAVING DATA
    ###############################################################################
    def test_save_data(self):
        exp_setup = load_cal_aero_setup()
        exp_setup.calculate_gt_trajectory_of_tag(16, time_horizon=0.1, max_std=0.1)
        save_path = "../ESP_code/data/Aerolytics_small_extract_GT"
        exp_setup.save_gt(save_path)
        exp_setup.plot_3D()
        plt.show()

    def test_load_gt_data(self):
        exp_setup = load_cal_aero_setup()
        load_path = "../ESP_code/data/Aerolytics_small_extract_GT"
        exp_setup.load_gt(load_path)
        exp_setup.plot_3D()
        plt.show()

    def test_load_vio_data(self):
        filepath = "../ESP_code/data/Long Experiment noon VIO/data.csv"
        exp_setup = load_cal_aero_setup()
        exp_setup.load_vio_data(filepath, 16)

    def test_load_imu_data(self):
        filepath = "../ESP_code/data/Long Experiment noon VIO/data_imu.csv"
        exp_setup = load_cal_aero_setup()
        exp_setup.load_imu_data(filepath, 16)
        print(exp_setup.imu_data)

    def test_save_range_data(self):
        exp_setup = load_cal_aero_setup(folder_path="../ESP_code/data/Long Experiment noon")
        save_path = "../ESP_code/data/Long Experiment noon Range/range_16_6"
        exp_setup.save_range_data(save_path, 6, 16)

if __name__ == '__main__':
    unittest.main()
