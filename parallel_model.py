from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time


DATA_PATH = '/home/lyn/policies/randomized_push_data/chosen_data'
SAVE_PATH = '/home/lyn/HitLyn/Push/saved_model_randomized'

def input_from_world_to_object(state):
    """turn a vector which is relative to world coordinate to object coordinate, change the input of the network!!!!
    Args:
        object_x: position x of object (in world coordinate)
        object_y: position y of object (in world coordinate)
        theta: object's rotation in anticlockwise (radians)
        robot_x: position x of robot (in world coordinate)
        robot_y: position y of robot (in world coordinate)
        robot_x_s: nextstep position x of robot (in world coordinate)
        robot_y_s: nextstep position y of robot (in world coordinate)
    Returns:
        (robot_x, robot_y, action_x, action_y): relative position and action of robot to object(in object coordinate)"""
    assert state.shape == (7,)
    object_x, object_y, theta, robot_x, robot_y, robot_x_s, robot_y_s = (i for i in state)
    delta_position_x, delta_position_y = robot_x - object_x, robot_y - object_y
    delta_action_x, delta_action_y = robot_x_s - robot_x, robot_y_s - robot_y
    robot_position_relative_to_object_x = delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    robot_position_relative_to_object_y = delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)
    action_relative_to_object_x = delta_action_x*np.cos(theta) + delta_action_y*np.sin(theta)
    action_relative_to_object_y = delta_action_y*np.cos(theta) - delta_action_x*np.sin(theta)

    return np.array([robot_position_relative_to_object_x, robot_position_relative_to_object_y, action_relative_to_object_x, action_relative_to_object_y])

def target_from_world_to_object(object_state_s, object_state):
    delta_position_x, delta_position_y, delta_theta = (i for i in (object_state_s - object_state))
    theta = object_state[2]
    delta_position_relative_to_object_x =  delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    delta_position_relative_to_object_y =  delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)

    return np.array([delta_position_relative_to_object_x, delta_position_relative_to_object_y, delta_theta])

def vel_from_world_to_object(object_state_current, object_state_before):
    delta_position_x, delta_position_y, delta_theta = (i for i in (object_state_current - object_state_before))
    theta = object_state_current[2]
    vel_relative_to_object_x = delta_position_x*np.cos(theta) + delta_position_y*np.sin(theta)
    vel_relative_to_object_y = delta_position_y*np.cos(theta) - delta_position_x*np.sin(theta)

    return np.array([vel_relative_to_object_x, vel_relative_to_object_y, delta_theta])

def get_sample_input(env_time_step, object_state, robot_state, episode, i):
    """
    Returns: np.array, size(time_steps, 4 + 3) 4: robot pos and act 3: vel of the object relative to object coordinate
        """
    # get robot pos and act
    idx = episode * env_time_step + i
    state = np.concatenate((object_state[idx], robot_state[idx]))
    action = robot_state[idx + 1]
    sample_input_robot = input_from_world_to_object(np.concatenate((state, action)))
    # get object velocity
    if i == 0:
        vel = np.zeros([3])
    else:
        vel = vel_from_world_to_object(object_state[idx], object_state[idx - 1])
    sample_input = np.concatenate([vel, sample_input_robot])

    return sample_input

def get_sample_target(env_time_step, object_state, episode, i):
    """object_state
    Returns: np.array, size(time_steps, 3)
        """

    idx = episode * env_time_step + i
        # state = (object_state[idx + 1] - object_state[idx])
    sample_targets = target_from_world_to_object(object_state[idx + 1], object_state[idx])

    return sample_targets

def stack_time_step_input(env_time_step, state, timestep, episode, i):
     sample_inputs = []

     for step in range(timestep):
         idx = episode * (env_time_step - 1) + i * timestep + step

         sample_inputs.append(state[idx])

     return np.array(sample_inputs)

def stack_time_step_output(env_time_step, state, timestep, episode, i):

     sample_inputs = []

     for step in range(timestep):
         idx = episode * (env_time_step - 1) + i * timestep + step

         sample_inputs.append(state[idx])

     return np.array(sample_inputs)

class Model():
    def __init__(self, object_state_dim, robot_action_dim, env_time_step, rnn_units, batch_size, time_steps, load_data = False, build = True):
        """
        Args:
            object_state_dim(int):
            robot_action_dim(int):
            env_time_step(int): the time steps we cut from the env(50)
            rnn_units(int):
            batch_size(int):
            time_steps(int): how many time steps we need to integrate as a sequence
        """
        # init #
        self.object_state_dim = object_state_dim
        self.robot_action_dim = robot_action_dim
        self.env_time_step = env_time_step
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.input_feature_size =2*self.robot_action_dim + self.object_state_dim # 7 in our env
        self.load_data = load_data

        self.epochs = 500
        self.build = build


        # model build #
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(rnn_units*2, return_sequences = True, recurrent_initializer='glorot_uniform', recurrent_regularizer=tf.keras.regularizers.l2(0.05), input_shape = [self.time_steps, self.input_feature_size], dropout = 0.5),
            # # tf.keras.layers.Dropout(0.8),
            tf.keras.layers.LSTM(rnn_units, return_sequences = True, recurrent_initializer='glorot_uniform', recurrent_regularizer=tf.keras.regularizers.l2(0.05), dropout = 0.5),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(object_state_dim, kernel_regularizer=tf.keras.regularizers.l2(0.05))
        ])

        # self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.losses.MeanAbsoluteError())
        # self.model.build((self.batch_size, None, self.input_feature_size))

        # data load and preprocess #
        if self.load_data:
            self.object_data = np.load(os.path.join(DATA_PATH, 'object_data.npy'))
            self.robot_data = np.load(os.path.join(DATA_PATH, 'robot_data.npy'))
            self.train_object_data, self.test_object_data, self.train_robot_data, self.test_robot_data = train_test_split(self.object_data, self.robot_data, test_size = 0.2)
            self.train_object_data, self.test_object_data, self.train_robot_data, self.test_robot_data = self.train_object_data.reshape(-1, 7), self.test_object_data.reshape(-1, 7), self.train_robot_data.reshape(-1, 3), self.test_robot_data.reshape(-1, 3)

            self.train_data_set = self.data_preprocess(self.train_object_data, self.train_robot_data)
            self.test_data_set = self.data_preprocess(self.test_object_data, self.test_robot_data)

        # self.evaluate_data_set = self.data_preprocess(self.evaluate_object_data, self.evaluate_robot_data)


        # optimizer and loss function
        self.loss = tf.losses.MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam()
        # metrics
        self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        self.train_absolute_error = tf.keras.metrics.MeanAbsoluteError(name = 'absolute_error')

        self.test_loss = tf.keras.metrics.Mean(name = 'test_loss')
        self.test_absolute_error = tf.keras.metrics.MeanAbsoluteError(name = 'test_error')

        if self.build:
            self.model.build(tf.TensorShape([1, None, self.input_feature_size]))
        #
        # self.evaluate_loss = tf.keras.metrics.Mean(name = 'evaluate_loss')
        # self.test_error = tf.keras.metrics.MeanAbsolutePercentageError(name = 'evaluate_error')
        # self.evaluate_absolute_error = tf.keras.metrics.MeanAbsoluteError(name = 'evaluate_error')


    def LSTM_train_step(self, input, target):
        with tf.GradientTape() as tape:
            predictions = self.model(input)
            loss = 1000*self.loss(target, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        # self.train_error(target, predictions)
        # self.train_absolute_error(target, predictions)

    def LSTM_test_step(self, input, target):
        predictions = self.model(input)
        loss = 1000*self.loss(target, predictions)
        self.test_loss(loss)
        # self.test_error(target, predictions)
        # self.test_absolute_error(target, predictions)



    def train(self, data_set, test_data_set):
        """
        Args:
            data_set(dict): data_set['input']: np.array, data_set['target']: np.array
        """

        dataset = tf.data.Dataset.from_tensor_slices((data_set['input'], data_set['target']))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data_set['input'], test_data_set['target'])).shuffle(1000000).batch(self.batch_size, drop_remainder = True)
        # evaluate_dataset = tf.data.Dataset.from_tensor_slices((evaluate_data_set['input'], evaluate_data_set['target'])).shuffle(1000000).batch(self.batch_size, drop_remainder = True)

        for epoch in range(self.epochs):
            dataset_epoch = dataset.shuffle(10000000).batch(self.batch_size, drop_remainder = True)

            for input, target in dataset_epoch:
                self.LSTM_train_step(input, target)

            for test_input, test_target in test_dataset:
                self.LSTM_test_step(test_input, test_target)
            #
            # for evaluate_input, evaluate_target in evaluate_dataset:
            #     self.LSTM_evaluate_step(evaluate_input, evaluate_target)

            template = 'Epoch {}, Train Loss: {}, Test Loss: {}'
            print(template.format(epoch + 1,
                                self.train_loss.result(),
                                # self.train_absolute_error.result(),
                                self.test_loss.result()))
                                # self.test_absolute_error.result()))

            # reset the metrics for next epoch
            self.train_loss.reset_states()
            # self.train_absolute_error.reset_states()
            # self.evaluate_loss.reset_states()
            self.test_loss.reset_states()
            # self.test_absolute_error.reset_states()

            #save weights
            if (epoch + 1) % 10 == 0:
                dirpath = os.path.join(SAVE_PATH, 'epoch' + str(epoch + 1), 'log')
                namepath = os.path.join(SAVE_PATH, 'epoch' + str(epoch + 1), '60_steps')
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                    print('saving weights to', namepath)
                    self.model.save_weights(namepath)
                else:
                    self.model.save_weights(namepath)



    def data_preprocess(self, object_data, robot_data):

        object_position = object_data[:, :2]
        object_rotation = (np.arccos(object_data[:, 3]) * 2).reshape(-1, 1)
        object_state = np.concatenate((object_position, object_rotation), axis = 1)

        assert object_state.shape == (len(object_position), 3)

        robot_state = robot_data[:, :2]
        assert robot_state.shape == (len(robot_state), 2)

        episode_num = len(object_state)//self.env_time_step # how many episodes in the dataset: 101 timestep each
        sequences_num_per_episode = (self.env_time_step - 1)//self.time_steps # how many sequences to generate per episode

        sample_inputs = np.zeros((episode_num * sequences_num_per_episode, self.time_steps, 7))
        sample_targets = np.zeros((episode_num * sequences_num_per_episode, self.time_steps, 3))

        state_inputs = np.zeros((episode_num * (self.env_time_step - 1), 7))
        state_targets = np.zeros((episode_num * (self.env_time_step - 1), 3))

        # get state
        for episode in range(episode_num):
            for i in range(self.env_time_step - 1):
                idx = episode*(self.env_time_step - 1) + i
                state_inputs[idx] = get_sample_input(self.env_time_step, object_state, robot_state, episode, i)
                state_targets[idx] = get_sample_target(self.env_time_step, object_state, episode, i)

        # get inputs and targets
        for episode in range(episode_num):
            for i in range(sequences_num_per_episode):
                idx = episode * sequences_num_per_episode + i
                sample_inputs[idx] = stack_time_step_input(self.env_time_step, state_inputs, self.time_steps, episode, i)
                sample_targets[idx] = stack_time_step_output(self.env_time_step, state_targets, self.time_steps, episode, i)

        data_set = {}
        data_set['input'] = sample_inputs
        data_set['target'] = sample_targets


        return data_set


    def predict(self, state):
        """
        Because of the way the RNN state is passed from timestep to timestep,
        the model only accepts a fixed batch size once built.
        To run the model with a different batch_size, we need to rebuild the model
        Args:
            state: np.array, size = (1, self.time_steps, 7)
        """
        # self.model.build(tf.TensorShape([1, None, self.input_feature_size]))
        output = self.model.predict(state)
        # print('LSTM processed')
        return output[:, -1, :]


    def evaluate(self, inputs, targets):
        pass

    def load_weights(self, path):
        self.model.load_weights(path)


def main():
    model = Model(3, 2, 42, 64, 64, 4, load_data = True)
    train_data_set = model.train_data_set
    test_data_set = model.test_data_set
    # evaluate_data_set = model.evaluate_data_set
    # print(data_set['input'][50], data_set['target'][50])
    # print(model.model_normal.summary())
    model.train(train_data_set, test_data_set)

if __name__ == '__main__':
    main()
