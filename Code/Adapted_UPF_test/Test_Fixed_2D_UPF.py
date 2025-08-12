import unittest
import numpy as np
from UPF_demo_code.Adapted_UPF.Fixed2DUPF import Fixed2DUPF, Fixed2DUPFDataLogger
from Code.Simulation.NLOS_Manager import NLOS_Manager
from Code.DataLoggers.ConnectedAgent_DataLogger import UPFConnectedAgentDataLogger
from Code.Simulation.BiRobotMovement import drone_flight, run_simulation, Control2D, fix_connected_2D_host, fix_host_fix_connected
from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix
import matplotlib.pyplot as plt
class MyTestCase(unittest.TestCase):
    def init_test(self, sigma_v=0.1, sigma_w=0.1, sigma_uwb=0.1, drifting_host=False):
        # Paper = Relative Transformation Estimation Based on Fusion of Odometry and UWB.py Ranging Data

        self.uwb_time_steps = 1000  # (120 // 0.03)          # Paper simulation time = 120s
        self.odom_time_step = 0.1
        self.uwb_time_step = 0.1  # Paper experiments UWB.py frequency = 37 Hz
        self.factor = int(self.uwb_time_step / self.odom_time_step)
        self.simulation_time_steps = self.uwb_time_steps * self.factor

        self.sigma_uwb = sigma_uwb  # Paper sigma uwb = 0.1
        self.sigma_v = sigma_v  # Paper sigma odom = 0.001 m -> not sure how this relates with the heading error.
        self.sigma_w = sigma_w  # / 180 * np.pi  # In the paper they use degrees.

        self.los = []
        self.drifting_host = drifting_host
        self.nlos_man = NLOS_Manager(nlos_bias=2.)
        self.debug = False

    def init_drones(self, x_ca_0, h_ca_0, max_range=None):
        self.max_range = max_range
        ha_pose_0 = np.array([0, 0, 0, 0])
        ca_pose_0 = np.concatenate([x_ca_0, np.array([h_ca_0])])
        self.drone = drone_flight(ca_pose_0, sigma_dv=self.sigma_v, sigma_dw=self.sigma_w, max_range=self.max_range,
                                  origin_bool=True, simulation_time_step=self.odom_time_step)
        self.host = drone_flight(ha_pose_0, sigma_dv=self.sigma_v, sigma_dw=self.sigma_w, max_range=self.max_range,
                                 origin_bool=True, simulation_time_step=self.odom_time_step)

        distance = np.linalg.norm(self.drone.x_start - self.host.x_start)
        self.startMeasurement = distance + np.random.randn(1) * self.sigma_uwb

    def run_test(self, nlos_function, name="Unidentified Test"):
        self.dl = Fixed2DUPFDataLogger(self.host, self.drone, self.ca)
        # self.ca.set_logging(self.dl)
        # self.dl.log_data(0)
        dx_ca = np.zeros(4)
        q = np.zeros((4, 4))
        q_ha = np.zeros((4, 4))
        for i in range(1, self.simulation_time_steps):
            print("Simulation step: ", i, " /", self.simulation_time_steps)
            # print(len(self.ca.particles))
            # try:
            d_dx_ca = np.concatenate((self.drone.dx_slam[i], np.array([self.drone.dh_slam[i]])))
            f = get_4d_rot_matrix(dx_ca[-1])
            dx_ca = dx_ca + f @ d_dx_ca
            q = q + f @ self.drone.q @ f.T
            q_ha = q_ha + self.host.q

            # if self.drifting_host:
            d_dx_ha = np.concatenate((self.host.dx_slam[i], np.array([self.host.dh_slam[i]])))
            self.ca.ha.predict(d_dx_ha, self.host.q)

            if i % self.factor == 0:
                distance = np.linalg.norm(self.drone.x_real[i] - self.host.x_real[i])
                uwb_measurement = distance + np.random.randn(1)[0] * self.sigma_uwb
                uwb_measurement, los_state = nlos_function(int(i / self.factor), uwb_measurement)
                self.los.append(los_state)

                if self.drifting_host:
                    x_ha = self.host.x_slam[i]
                    h_ha = self.host.h_slam[i]
                    x_ha = np.concatenate([x_ha, np.array([h_ha])])
                else:
                    x_ha = self.host.x_real[i]
                    h_ha = self.host.h_real[i]
                    x_ha = np.concatenate([x_ha, np.array([h_ha])])

                self.ca.ha.update(x_ha, q_ha)
                self.ca.run_2d_model(uwb_measurement)

                self.dl.log_data(i)

                dx_ca = np.zeros(4)
                q = np.zeros((4, 4))
                self.ca.ha.reset_integration()
                # if not self.drifting_host:
                q_ha = np.zeros((4, 4))
            # except Exception as e:
            #     print(e)
            #     break
            print(self.ca.best_particle.t_si_sj)

    def test_tc1(self):
        # Length of NLOS  is proportional to error on odom?
        z= 0
        self.init_test(sigma_v=0.01, sigma_w=0.001, sigma_uwb=0.1,
                       drifting_host=True)
        self.init_drones(np.array([2, 0, z]), 0, max_range=1)
        c2d = Control2D(self.host)
        c2d.set_boundries(radius=1)
        run_simulation(self.simulation_time_steps, self.host, self.drone,
                       fix_connected_2D_host, kwargs={"control2d": c2d})
        # run_simulation(self.simulation_time_steps, self.host, self.drone,
                       # fix_host_fix_connected, kwargs={"control2d": c2d})
        self.ca = Fixed2DUPF("0x000", x_ha_0=np.concatenate((self.host.x_start, [self.host.h_start])))
        self.ca.set_ukf_parameters(kappa=-1, alpha=1, beta=2)
        self.ca.set_initalisation_parameters(self.sigma_uwb, 12, 1, z, 0.1)
        # self.ca.split_sphere_in_equal_areas(self.startMeasurement[0], 2*self.sigma_uwb, n_azimuth=12, n_heading=1,
        #                                     dz=z, dz_sigma=0.1)

        self.run_test(nlos_function=self.nlos_man.los)

        self.dl.plot_self(self.los)
        # self.dl.get_best_particle_log().create_3d_plot()
        self.dl.get_best_particle_log().plot_error_graph()
        self.dl.get_best_particle_log().plot_ukf_states()
        self.dl.get_best_particle_log().plot_2D_drift()
        plt.show()


if __name__ == '__main__':
    unittest.main()
