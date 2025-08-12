import unittest
import numpy as np
from UPF_demo_code.Adapted_UPF.Fixed2DUPF import Fixed2DUPF, Fixed2DUPFDataLogger
from Code.Simulation.NLOS_Manager import NLOS_Manager
from Code.DataLoggers.ConnectedAgent_DataLogger import UPFConnectedAgentDataLogger
from Code.Simulation.BiRobotMovement import drone_flight, run_simulation, Control2D, fix_connected_2D_host, fix_host_fix_connected
from Code.UtilityCode.utility_fuctions import get_4d_rot_matrix
import matplotlib.pyplot as plt

from UPF_demo_code.ITF_husky.LidarOdom import LIDAROdom

import pickle

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

    def load_data(self, file= "./data/RPE_debug/standstill.pkl"):
        self.data = None
        with open(file, 'rb') as f:
            self.data = pickle.load(f)
        self.data["dt"] = []
        self.data["d_t_odom"] = []
        self.data["t_odom"] = []
        self.data["t_rpe_cal"] = []
        print(len(self.data["p"]), len(self.data["q"]), len(self.data["t"]), len(self.data["rpe"]))
        self.simulation_time_steps = len(self.data["p"])
        # self.simulation_time_steps = 10



    def run_test(self, nlos_function, name="Unidentified Test"):
        # self.dl = Fixed2DUPFDataLogger(self.host, self.drone, self.ca)
        q_ha = np.zeros((4, 4))
        self.lo = LIDAROdom()
        sigma_x = (self.sigma_v)**2
        sigma_h = (self.sigma_w)**2
        self.lo.set_q_dt(np.array([[sigma_x, 0, 0, 0],
                                  [0,sigma_x, 0, 0],
                                  [0, 0, sigma_x, 0],
                                  [0, 0, 0, sigma_h]],))

        for i in range(self.simulation_time_steps):
            print("Simulation step: ", i, " /", self.simulation_time_steps)
            # q = np.array([self.data["q"][i][1], self.data["q"][i][2], self.data["q"][i][3], self.data["q"][i][0]])
            self.lo.odom_callback(self.data["p"][i], self.data["q"][i], self.data["t"][i])
            self.data["dt"].append(self.lo.dt)
            self.data["d_t_odom"].append(self.lo.d_t_odom)
            self.data["t_odom"].append(self.lo.t_odom)

            print("p: ", self.data["p"][i], "; q: ",self.data["q"][i],"; t: ", self.data["t"][i], "; d: ", self.data["d"][i])
            print("dt: ", self.data["dt"][i], "; d_t_odom: ",self.data["d_t_odom"][i],  "; t_odom: ", self.data["t_odom"][i])
            print("q: ", self.lo.q)

            self.ca.ha.predict(self.lo.d_t_odom, self.lo.q)

            if i % self.factor == 0:
                uwb_measurement = self.data["d"][i]
                _, los_state = nlos_function(int(i / self.factor), uwb_measurement)
                self.los.append(los_state)

                self.ca.ha.update(self.lo.t_odom, self.lo.q)
                self.ca.run_2d_model(uwb_measurement)
                print("==========RPE===========")
                par_t = []
                for i, particle in enumerate(self.ca.particles):
                    par_t.append(particle.t_si_sj)
                    print(i, particle.t_si_sj)
                self.data["t_rpe_cal"].append(par_t)



                self.ca.ha.reset_integration()
                print(self.ca.best_particle.t_si_sj)

    def test_tc1(self):
        # Length of NLOS  is proportional to error on odom?
        z= 0
        self.init_test(sigma_v=0.05, sigma_w=0.001, sigma_uwb=0.1,
                       drifting_host=True)

        self.load_data("./data/RPE_debug/drift_issue.pkl")


        self.ca = Fixed2DUPF("0x000", x_ha_0=np.zeros(4))
        self.ca.set_ukf_parameters(kappa=-1, alpha=1, beta=2)
        self.ca.set_initalisation_parameters(self.sigma_uwb, 12, 1, z, 0.1)
        # self.ca.split_sphere_in_equal_areas(self.data[0], 2*self.sigma_uwb, n_azimuth=12, n_heading=1,
                                            # dz=z, dz_sigma=0.1)

        self.run_test(nlos_function=self.nlos_man.los)


        # plt.show()


if __name__ == '__main__':
    mt = MyTestCase()
    mt.test_tc1()
    # mt.load_data("./data/RPE_debug/drift_issue.pkl")
    p_odom = np.array(mt.data["p"])
    dt_odom = np.array(mt.data["d_t_odom"])
    best_rpe = []
    for rpe in mt.data["t_rpe_cal"]:
        best_rpe.append(rpe[0])
    best_rpe = np.array(best_rpe)

    #plot the first 3 colums of mt.data["t_odom"]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(p_odom[:,0], p_odom[:,1], p_odom[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    # fig = plt.figure()
    fig, ax = plt.subplots(4,1)
    ax[0].plot(dt_odom[:,0])
    ax[1].plot(dt_odom[:,1])
    ax[2].plot(dt_odom[:,2])
    ax[-1].plot(mt.data["d"])
    plt.show()

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(best_rpe[:, 0])
    ax[1].plot(best_rpe[:, 1])
    ax[2].plot(best_rpe[:, 2])
    # ax[-1].plot(mt.data["d"])
    plt.show()


    # unittest.main()
# Path: UPF_demo_code/Adapted_UPF_test/Test_Fixed_2D_UPF.p
