import unittest
import Experimental_Setup
import matplotlib.pyplot as plt

class MyTestCase(unittest.TestCase):

    def load_setup(self):
        rough_positions = [[0, 4.5 - 0.86, 3],
                           [0, 4.5 - 0.14, 0.45],
                           [0, 0.94, 3],
                           [0, 0.1, 0.43],
                           [32, 4.5 - 1.35, 3],
                           [8 - 3.3, 0.99, 3],
                           [8, 4.5 - 0.025, 1.65],
                           [8, -0.87, 3],
                           [8 + 4.14, 1, 0.97, 3],
                           [8 + 4.14, 4.5 - 1.3, 3],
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
        return exp_setup

    ##################################################################################
    #### LOADING
    ##################################################################################

    def test_read_data(self):
        exp_setup = Experimental_Setup.Experiment()
        folder_path = "../ESP_code/data/udp_data_2025-09-18_11-13-41"
        exp_setup.load_data(folder_path)
        print(exp_setup.odom_data)

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
                           [8 - 3.3, 0.99, 3],
                           [8, 4.5 - 0.025, 1.65],
                           [8, -0.87, 3],
                           [8 + 4.14, 1, 0.97, 3],
                           [8 + 4.14, 4.5 - 1.3, 3],
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
        exp_setup = self.load_setup()
        exp_setup.get_anchor_ranges_statistics()

    def test_anchor_cal(self):
        exp_setup = self.load_setup()
        exp_setup.calibrate_anchor_pos()

    ##################################################################################
    #### PLOTTING
    ##################################################################################
    def test_plot_range(self):
        exp_setup = self.load_setup()
        exp_setup.plot_range(1, 16, plot_rssi=True)
        plt.show()

    def test_plot_ranges(self):
        exp_setup = self.load_setup()
        exp_setup.plot_ranges(separate_plots=False)
        plt.show()

if __name__ == '__main__':
    unittest.main()
