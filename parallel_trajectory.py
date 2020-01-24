from __future__ import print_function
from __future__ import division


import numpy as np
import copy

def vel_from_world_to_object(object_state_current, object_state_before):
    delta_position_x, delta_position_y, delta_theta = (i for i in (object_state_current - object_state_before))
    theta = object_state_current[2]
    vel_relative_to_object_x = delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    vel_relative_to_object_y = delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)

    return np.array([vel_relative_to_object_x, vel_relative_to_object_y, delta_theta])

def robot_pos_from_world_to_relative(object_pos, robot_pos):

    object_x, object_y= [object_pos[i] for i in range(2)]
    theta = 2*np.arccos(object_pos[6])
    robot_x, robot_y = [robot_pos[i] for i in range(2)]

    delta_position_x, delta_position_y = robot_x - object_x, robot_y - object_y
    robot_position_relative_to_object_x = delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    robot_position_relative_to_object_y = delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)

    return np.array([robot_position_relative_to_object_x, robot_position_relative_to_object_y])

def robot_action_from_world_to_object(action, object_pos):
    action_x = action[0]
    action_y = action[1]
    theta = object_pos[2]
    action_relative_to_object_x = action_x*np.cos(theta) + action_y*np.sin(theta)
    action_relative_to_object_y = action_y*np.cos(theta) - action_x*np.sin(theta)
    return np.array([action_relative_to_object_x, action_relative_to_object_y])

class Trajectory():
    """collect trajectory history and preprocess the data making it more suitable for the input of predictor"""
    def __init__(self):
        self.goal = np.zeros([3])
        self.obstacle_pos = np.zeros([3])
        self.relative_state = [] # list of numpy.array(delta_x,delta_y,delta_theta,x_robot, y_robot, action_x, action_y)
        self.real_robot_action = [] # list of numpy.array(action_x_world, action_y_world)
        self.absolute_state = [] # list of numpy.array(x_object,y_object,theta_object,x_robot,y_robot)
        self.object_absolute_state = [] # list of numpy.array(pos_x, pos_y, theta)
        self.robot_absolute_state = [] # list of numpy.array(pos_x, pos_y)
        self.count = 0


    def get_relative_state(self):
        relative_state = copy.copy(self.relative_state)
        return np.asarray(relative_state)

    def get_absolute_state(self):
        absolute_state = copy.copy(self.absolute_state)
        return np.asarray(absolute_state)

    def get_obstacle_position(self):
        obstacle_pos = copy.copy(self.obstacle_pos)
        return np.array([obstacle_pos[0], obstacle_pos[1]])

    def get_goal_position(self):
        return self.goal

    def set_goal(self, pos_x, pos_y, theta):
        self.goal[0] = pos_x
        self.goal[1] = pos_y
        self.goal[2] = theta

    def reset(self):
        # reset the count num
        self.count = 0

        del self.relative_state[:]
        del self.absolute_state[:]
        del self.object_absolute_state[:]
        del self.robot_absolute_state[:]
        del self.real_robot_action[:]

        self.goal = np.zeros([3])
        self.obstacle_pos = np.zeros([3])

    def update_state(self, pose_object, pose_tool):
        robot_pos = copy.copy(pose_tool) # absolute
        object_pos = copy.copy(pose_object) # absolute
        self.object_absolute_state.append(np.array([object_pos[0], object_pos[1], 2*np.arccos(object_pos[6])]))
        self.robot_absolute_state.append(robot_pos)

        # updata relative_state
        if self.count == 0:
            object_vel = np.zeros([3])
        else:
            object_current_state = copy.copy(self.object_absolute_state[self.count])
            object_previous_state = copy.copy(self.object_absolute_state[self.count - 1])
            object_vel = vel_from_world_to_object(object_current_state, object_previous_state)

        robot_pos_relative = robot_pos_from_world_to_relative(object_pos, robot_pos)
        robot_action_relative = np.zeros([2])
        self.relative_state.append(np.concatenate([object_vel, robot_pos_relative, robot_action_relative]))

        # update absolute state
        self.absolute_state.append(np.concatenate([self.object_absolute_state[-1], self.robot_absolute_state[-1]]))

        # update count
        self.count += 1

    def update_action(self, action):
        """update the real robot action list both in"""
        action_ = copy.copy(action)
        self.real_robot_action.append(action_)
        object_pos = copy.copy(self.object_absolute_state[self.count - 1])
        action_relative_to_object = robot_action_from_world_to_object(action_, object_pos)
        self.relative_state[self.count - 1][5:] = action_relative_to_object
