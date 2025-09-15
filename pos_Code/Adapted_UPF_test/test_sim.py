import unittest

from Code.Simulation.RobotClass import NewRobot
from Code.UtilityCode.Measurement import create_experimental_data
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pos_Code.Adapted_UPF.Fixed2DUPF import Fixed2DUPF, Fixed2DUPFDataLogger


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def reset_agents_w_exp_data(self, experiment_data):
        self.parameters={}
        self.agents = {}
        for drone_name in experiment_data["drones"]:
            T_vicon = experiment_data["drones"][drone_name]["T_real"]
            DT_vio = experiment_data["drones"][drone_name]["DT_slam"]
            Q_vio = experiment_data["drones"][drone_name]["Q_slam"]

            drone = NewRobot()
            drone.from_experimental_data(T_vicon, DT_vio, Q_vio, experiment_data["sample_freq"])
            self.agents[drone_name] = {"drone": drone}
        self.d0 = experiment_data["uwb"][0]
        self.experiment_data = experiment_data


    def run_exp(self, test_name="test"):
        self.los_state = []
        drone0: NewRobot = self.agents["drone_0"]["drone"]
        drone1: NewRobot = self.agents["drone_1"]["drone"]

        # drone0.form_experimental_data(self.experiment_data["drone_0"], self.experiment_data["Q_vio"])
        # drone1.set_vio_slam(self.experiment_data["drone_1"], self.experiment_data["Q_vio"])
        distances = self.experiment_data["uwb"]
        exp_len = len(self.experiment_data["uwb"])
        self.factor = int(self.experiment_data["sample_freq"] / self.frequency)
        for i in range(0, exp_len - 1):
            if self.debug_bool:
                print(datetime.now(), " Experiment step: ", i, " /", exp_len)
            if self.plot_bool:
                pass
                # self.sim.plot_trajectories_evolution(i, 50)

            # Integrate the odometry:
            for drone in self.agents:
                self.agents[drone]["drone"].integrate_odometry(i)

            if i % self.factor == 0:
                uwb_measurement = np.linalg.norm(drone0.x_real[i] - drone1.x_real[i])
                self.los_state.append(self.experiment_data["los_state"][i])
                if np.isnan(uwb_measurement):
                    print("UWB measurement is NaN at step", i)
                else:
                    dx_0, q_0 = drone0.reset_integration()
                    dx_1, q_1 = drone1.reset_integration()
                    if self.bool_2D:
                        dx_0[2] = 0
                        dx_1[2] = 0
                        q_0[2,2]  = 1e-8
                        q_1[2,2]  = 1e-8
                    x_ha = drone0.x_slam[i]
                    h_ha = drone0.h_slam[i]
                    x_ha_0 = np.concatenate([x_ha, np.array([h_ha])])
                    self.ca.ha.update(x_ha_0, q_0)
                    self.ca.run_2d_model(uwb_measurement)
                    self.dl.log_data(i)
                    self.los_state.append(1)

                    for particle in self.ca.particles:
                        print("Particle weight:", particle.weight, particle.t_si_sj)
                # Timing the execution of the algorihtm

        if self.plot_bool:
            plt.close()
        # eval("self.end_" + self.method + "_test()")

    def set_2d_upf(self):
        drone0: NewRobot = self.agents["drone_0"]["drone"]
        drone1: NewRobot = self.agents["drone_1"]["drone"]
        self.ca = Fixed2DUPF("0x000", x_ha_0=np.concatenate((drone0.x_slam[0], [drone0.h_slam[0]])))
        self.ca.set_ukf_parameters(kappa=-1, alpha=1, beta=2)
        self.ca.set_initalisation_parameters(0.1, 12, 1, -0.13, 0.1)
        self.frequency = 10
        self.dl = Fixed2DUPFDataLogger(drone0, drone1, self.ca)
        self.plot_bool = False
        self.debug_bool = True
        self.bool_2D =True
        self.los_state = []




    def test_sim(self):
        data_folder = "./sim_data/Guilaume_expodom_test_simu_aug_sampled.pkl"
        experiment_data, measurements = create_experimental_data(data_folder, 0.1, 0.1, 0.1, uwb_mu=0.0)
        self.reset_agents_w_exp_data(experiment_data[0])
        self.set_2d_upf()


        self.run_exp()

        self.dl.plot_self(self.los_state)
        # self.dl.get_best_particle_log().create_3d_plot()
        self.dl.get_best_particle_log().plot_error_graph()
        self.dl.get_best_particle_log().plot_ukf_states()
        self.dl.get_best_particle_log().plot_2D_drift()
        plt.show()

if __name__ == '__main__':
    unittest.main()
