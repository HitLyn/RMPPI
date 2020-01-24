from __future__ import print_function
from __future__ import division
import numpy as np
import os
import copy

from parallel_predictor import Predictor
from parallel_trajectory import Trajectory

class MPPI():
    """MPPI algorithm for pushing """
    def __init__(self, K, T, A):
        self.A = A # action scale
        self.T = T # time steps per sequence
        self.K = K # sample actions

        self.predictor = Predictor(self.K, self.T)
        self.trajectory = Trajectory()
        self.trajectory.reset()
        self.lambd = 1
        self.dim_u = 1 # U is theta

        # action init
        self.U_reset()
        self.u_init = np.array([0.0])
        self.cost = np.zeros([self.K])
        self.noise = np.zeros([self.K, self.T, self.dim_u])

    def compute_cost(self, step):
        action_list = np.zeros([self.K, self.T, 3])
        self.noise = np.clip(np.random.normal(loc = 0, scale = 1, size = (self.K, self.T, self.dim_u)), -1.57, 1.57)
        self.predictor.catch_up(self.trajectory.get_relative_state(), self.trajectory.get_absolute_state(), self.trajectory.get_goal_position(), step)
        eps = copy.copy(self.noise)

        # compute for T time steps
        for t in range(self.T):
            if t > 0:
                eps[:, t] = 0.8*eps[:, t - 1] + 0.2*eps[:, t]
            self.noise[:, t] = copy.copy(eps[:, t])
            theta = self.U[t] + eps[:, t] #(K, 1)
            assert theta.shape == (self.K, 1)
            action = copy.copy(self.A * np.concatenate([np.cos(theta), np.sin(theta)]， axis = 1)) # (K, 2)
            assert action.shape == (self.K, 2)
            cost = self.predictor.predict(action, step)
            assert cost.shape == (self.K,)
            self.cost += cost #(K,)

        action_list = copy.copy(self.predictor.get_sample_action_list()) #(K, T, 2)
        assert action_list.shape == (self.K, self.T, 2)
        color_list = copy.copy(self.cost)
        color_list_extension = ((1 - (color_list - np.min(color_list))/(np.max(color_list) - np.min(color_list))).reshape(self.K, 1)*np.ones([self.K, self.T])).reshape(self.K, self.T, 1)
        assert color_list_extension.shape == (self.K, self.T, 1)
        action_list = np.concatenate([action_list, color_list_extension], axis = -1) #(K, T, 3)
        return action_list

    def trajectory_clear(self):
        self.trajectory.reset()

    def trajectory_set_goal(self, pos_x, pos_y, theta):
        self.trajectory.set_goal(pos_x, pos_y, theta)

    def trajectory_update_state(self, pose_object, pose_tool):
        self.trajectory.update_state(pose_object, pose_tool)

    def trajectory_update_action(self, action):
        self.trajectory.update_action(action)

    def compute_noise_action(self):
        beta = np.min(self.cost)
        eta = np.sum(np.exp((-1/self.lambd) * (self.cost - beta))) + 1e-6
        w = (1/eta) * np.exp((-1/self.lambd) * (self.cost - beta))

        self.U += [np.dot(w, self.noise[:, t]) for t in range(self.T)]
        # print(self.U)
        theta_list = self.U
        action_x_list = np.cos(theta_list)
        action_y_list = np.sin(theta_list)
        action_list = copy.copy(self.A*np.concatenate([action_x_list, action_y_list], axis = 1))
        action = copy.copy(action_list[0])

        return action_list, action


    def U_reset(self):
        self.U = np.zeros([self.T, self.dim_u])

    def get_K(self):
        return self.K

    def U_update(self):
        self.U = np.roll(self.U, -1, axis = 0)
        self.U[-1] = self.u_init

    def cost_clear(self):
        self.cost = np.zeros([self.K])
