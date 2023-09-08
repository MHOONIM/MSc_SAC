# Test basic movement : Collision Avoidance
# RL Framework: Policy Gradient --> Soft Actor-Critic (SAC)
# Image + Distance sensors as the states
# Saved model: trained_SAC_PI_05.h5, trained_SAC_Q_05.h5, trained_SAC_V_05.h5
# Action spaces: Discrete --> 4 Dimensions [Left, Forward, Right, Backward]
# Note: 5th implementation of the SAC.
# Trained episode: 0


import airsim  # Import airsim API
import keras.layers
# import pprint
# import cv2
import numpy as np
import math
from random import random, randint, choice, randrange
from time import sleep

# ******************************************* Keras, Tensorflow library declaration ************************************
import tensorflow as tf
from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Input, MaxPool2D, Flatten, Concatenate
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
# import keras.backend as k
from tensorflow import gather_nd
from keras.losses import mean_squared_error
# ******************************************* Keras, Tensorflow library declaration ************************************


# ****************************************** Soft Actor-Critic class start *********************************************
class SACLearning:
    def __init__(self):
        # Initialise Airsim Client
        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.initial_pose = self.takeoff(2)  # Get initial pose of the drone (Will be used for starting the new episode)

        # Initialise Map
        self.map_x = 100
        self.map_y = 100
        self.min_depth_meter = 0
        self.max_depth_meter = 50

        # Initialise State Parameters
        self.death_flag = False
        self.terminated = False
        self.buffer_size = 102400  # Experience Replay Buffer size
        self.batch_buffer_size = 128  # Sampling batch size
        self.actions = [0, 1, 2, 3]  # Discrete action space
        self.number_distance_sensor = 8
        self.distance_target_dimension = 1  # <-- [x, y]
        # Current state parameters
        self.state = np.zeros([self.number_distance_sensor], dtype=float)
        self.distance_target = np.zeros([self.distance_target_dimension], dtype=float)
        # Next state parameters
        self.next_state = self.state  # S_{t+1}
        self.next_distance_target = np.zeros([self.distance_target_dimension], dtype=float)
        self.reward = 0  # R_{t+1}
        self.max_reward = 100
        self.min_reward = -100
        self.step = 0  # Counter for storing the experiences
        self.img_width = 84
        self.img_height = 84
        self.img_depth = 1  # <-- Image data type (1 = Black and White), (3 = RGB)
        self.append_reward = []
        self.prev_dis = 0

        # Get the initial latitude and longitude
        self.lat0 = self.drone.getMultirotorState().gps_location.latitude  # Get the initial latitude
        self.lon0 = self.drone.getMultirotorState().gps_location.longitude  # Get the initial longitude
        self.alt0 = self.drone.getMultirotorState().gps_location.altitude  # Get the initial altitude
        self.drone_coordinates = self.coordinates()  # Current coordinates of the drone
        self.next_drone_coordinates = self.drone_coordinates  # Next coordinates of the drone

        # Initialise the Target position
        self.final_coordinates = [100, 0]
        self.allowed_y = 5
        self.max_dis = 0
        self.progress = 0

        # Create Experience Replay Buffers [S_t, a_t, death_flag, R_{t+1}, S_{t+1}]
        self.current_state_sens = np.zeros([self.buffer_size, self.number_distance_sensor], dtype=float)
        self.p_action_data = np.zeros([self.buffer_size], dtype=float)
        self.action_data = np.zeros([self.buffer_size], dtype=float)
        self.death_flag_data = np.zeros([self.buffer_size], dtype=bool)
        self.reward_data = np.zeros([self.buffer_size], dtype=float)
        self.next_state_sens = np.zeros([self.buffer_size, self.number_distance_sensor], dtype=float)

        # Load the Saved Experienced Replay Buffer (erb)
        # loaded_erb = np.load('ERB_SAC_05.npz')
        # self.current_state_img = loaded_erb['arr_0']
        # self.current_state_sens = loaded_erb['arr_1']
        # self.p_action_data = loaded_erb['arr_2']
        # self.action_data = loaded_erb['arr_3']
        # self.death_flag_data = loaded_erb['arr_4']
        # self.reward_data = loaded_erb['arr_5']
        # self.next_state_img = loaded_erb['arr_6']
        # self.next_state_sens = loaded_erb['arr_7']

        # Initialise Networks
        # If it is the first time running --> create new models
        self.actor, self.action_value, self.state_value, self.tar_state_value = self.network_creation()
        self.tar_state_value.set_weights(self.state_value.get_weights())
        # If it is not --> loaded the trained models
        # self.actor = load_model('Trained_Models/SAC_01/trained_SAC_1300.h5')
        # self.action_value = load_model('Trained_Models/SAC_01/trained_SAC_Q_01.h5')
        # self.state_value = load_model('Trained_Models/SAC_01/trained_SAC_V_01.h5')
        # self.tar_state_value = load_model('Trained_Models/SAC_01/trained_SAC_V_01.h5')
        # self.epsilon_t = np.zeros([self.batch_buffer_size, len(self.actions)], dtype=float)
        self.actor_grad_glob_norm = 0
        self.state_value_grad_glob_norm = 0
        self.action_value_grad_glob_norm = 0
        # Updating flags
        self.update_count = 0  # Counter For Updating The Target Network
        self.update_period = 0  # Period For The Counter
        self.full_flag = False
        self.fit_step = 128  # Defined Period For Fit The Prediction Network
        self.training_record = 1
        self.epoch = 1

    # Taking Off Method
    def takeoff(self, delay):
        self.drone.takeoffAsync().join()
        sleep(delay)
        return self.drone.simGetVehiclePose()

    # Network Creation Method
    def network_creation(self):
        # Input_2: Distance_sensor_input (self.distance_sensors)
        distance_inputs = Input(shape=self.number_distance_sensor,)
        distance_dense_1 = Dense(128)(distance_inputs)
        distance_dense_2 = Dense(128)(distance_dense_1)
        distance_dense_3 = Dense(128)(distance_dense_2)
        distance_dense_4 = Dense(128)(distance_dense_3)
        distance_activation_1 = Activation('relu')(distance_dense_4)

        # Output layers
        policy_output = Dense(len(self.actions),
                              activation='softmax',
                              kernel_initializer=tf.keras.initializers.random_uniform(minval=-0.0003, maxval=0.0003))(distance_activation_1)
        action_value_output = Dense(len(self.actions), activation='linear')(distance_activation_1)
        state_value_output = Dense(1, activation='linear')(distance_activation_1)

        # Define the policy model (\pi)
        policy = Model(inputs=distance_inputs, outputs=policy_output, name='policy_model')
        policy.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1))
        # Define the action-value model (Q)
        action_value = Model(inputs=distance_inputs, outputs=action_value_output, name='action_value_model')
        action_value.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1))
        # Define the state-value model (V)
        state_value = Model(inputs=distance_inputs, outputs=state_value_output, name='state_value_model')
        state_value.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1))
        # Define the target state-value model (V')
        target_state_value = Model(inputs=distance_inputs, outputs=state_value_output, name='target_state_value_model')
        target_state_value.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1))

        return policy, action_value, state_value, target_state_value

    # Action Space Method
    def action_space(self, action_no):
        # Define speed and duration
        throttle = 1  # m/s
        duration = 1  # s
        # Lift compensator to keep the drone in the air
        if self.drone.getMultirotorState().gps_location.altitude < 123.66:
            lift_compensation = -0.25
        elif self.drone.getMultirotorState().gps_location.altitude > 125:
            lift_compensation = 0.25
        else:
            lift_compensation = 0
        # Action Space
        if action_no == 0:
            # Forward --> move +x
            self.drone.moveByVelocityAsync(throttle, 0, lift_compensation, duration).join()
        elif action_no == 1:
            # Left --> move -y
            self.drone.moveByVelocityAsync(0, -throttle, lift_compensation, duration).join()
        elif action_no == 2:
            # Right --> move +y
            self.drone.moveByVelocityAsync(0, throttle, lift_compensation, duration).join()
        elif action_no == 3:
            # Backward --> move -x
            self.drone.moveByVelocityAsync(-throttle, 0, lift_compensation, duration).join()
        else:
            # No action
            self.drone.moveByVelocityAsync(0, 0, 0, duration).join()

    # Environment Method ------ Action Taking and Reward Shaping
    def environment(self, action):
        # Take action
        self.action_space(action)

        # Get the drone coordinates after taking and action (next coordinates)
        self.next_drone_coordinates = self.coordinates()

        # Progression Reward
        # Finding the euclidean distance between object and the drone.
        dis = np.sqrt((self.final_coordinates[0] - self.next_drone_coordinates[0]) ** 2 +
                      (self.final_coordinates[1] - self.next_drone_coordinates[1]) ** 2)
        progress_reward = 100 - (100 * dis / self.max_dis)  # The progression in percent

        # Travel distance is always going to be positive value.
        travel_dis = np.sqrt((self.next_drone_coordinates[0] - self.drone_coordinates[0]) ** 2 +
                             (self.next_drone_coordinates[1] - self.drone_coordinates[1]) ** 2)

        # Check the previous distance and the current distance between the drone and the target
        if dis < self.prev_dis - 0.1:
            travel_dis = travel_dis * 10
        else:
            travel_dis = travel_dis * (-10)

        # Total reward
        r = travel_dis

        # Check if the drone is moving out of the map
        if self.next_drone_coordinates[0] < -0.1 or self.next_drone_coordinates[1] < -0.1:
            r = self.min_reward
            self.death_flag = True

        # Check The Collision (If true --> Dead - Penalised)
        if self.next_drone_coordinates[0] > 0.5:
            if self.drone.simGetCollisionInfo().has_collided:
                r = self.min_reward
                self.death_flag = True

        # Check The Destination (If true --> Terminated - Rewarded)
        if self.next_drone_coordinates[0] > self.final_coordinates[0]:
            self.death_flag = True
            if self.final_coordinates[1] - self.allowed_y <= self.next_drone_coordinates[1] \
                    <= self.final_coordinates[1] + self.allowed_y:
                r = self.max_reward
            else:
                r = r

        # # Normalise the reward
        # r = r / 100

        # Get The Distance Sensors Data (Next state S_{t+1})
        self.drone.getDistanceSensorData()
        self.next_state = [self.drone.getDistanceSensorData("DistanceWest").distance,
                           self.drone.getDistanceSensorData("DistanceNorthWest").distance,
                           self.drone.getDistanceSensorData("DistanceNorth").distance,
                           self.drone.getDistanceSensorData("DistanceNorthEast").distance,
                           self.drone.getDistanceSensorData("DistanceEast").distance,
                           self.drone.getDistanceSensorData("DistanceSouthEast").distance,
                           self.drone.getDistanceSensorData("DistanceSouth").distance,
                           self.drone.getDistanceSensorData("DistanceSouthWest").distance]

        # Get the next difference distance between the target and the drone.
        self.next_distance_target = progress_reward
        self.prev_dis = dis
        return r

    # Coordinate Conversion Method
    def coordinates(self):
        lat = self.drone.getMultirotorState().gps_location.latitude
        lon = self.drone.getMultirotorState().gps_location.longitude
        lat0rad = math.radians(self.lat0)
        mdeg_lon = (111415.13 * np.cos(lat0rad) - 94.55 * np.cos(3 * lat0rad) - 0.12 * np.cos(5 * lat0rad))
        mdeg_lat = (111132.09 - 566.05 * np.cos(2 * lat0rad) + 1.2 * np.cos(4 * lat0rad) - 0.002 * np.cos(6 * lat0rad))

        x = (lat - self.lat0) * mdeg_lat
        y = (lon - self.lon0) * mdeg_lon

        return [x, y]

    # ------------------------------------------- Networks Training Start ----------------------------------------------
    # Network Training Method
    def network_training(self, indices):
        # Train the network by fit the new input and output features
        # New input features = New S_t's images from the buffer
        # New target features = New Q-values defined by the Deep Q-Learning algorithm
        gamma = tf.constant(0.99)  # Discount factor

        # Initialise Sampling Batch of Experienced Replay Buffers [S_t, a_t, death_flag, R_{t+1}, S_{t+1}]
        current_state_distance_batch = np.zeros([self.batch_buffer_size, self.number_distance_sensor], dtype=float)
        next_state_distance_batch = np.zeros([self.batch_buffer_size, self.number_distance_sensor], dtype=float)
        reward_batch = np.zeros([self.batch_buffer_size], dtype=float)
        action_append = np.zeros([self.batch_buffer_size], dtype=int)
        prob_selected_action_batch = np.zeros([self.batch_buffer_size], dtype=float)

        # Store the experiences in the batch experienced replay buffers.
        for j in range(self.batch_buffer_size):
            current_state_distance_batch[j] = self.current_state_sens[indices[j]]
            next_state_distance_batch[j] = self.next_state_sens[indices[j]]
            reward_batch[j] = self.reward_data[indices[j]]
            action_append[j] = self.action_data[indices[j]]
            prob_selected_action_batch[j] = self.p_action_data[indices[j]]

        # ********************************** Find the gradient of the cost function ************************************
        # Prepare the state data
        # Convert the numpy array to tensor format in order to compute the gradient in the tensorflow's gradient tape
        current_state_distance_batch_tensor = tf.convert_to_tensor(current_state_distance_batch, dtype=tf.float32)
        next_state_distance_batch_tensor = tf.convert_to_tensor(next_state_distance_batch, dtype=tf.float32)
        reward_batch_tensor = tf.convert_to_tensor(reward_batch, dtype=tf.float32)

        # Slice predicted Q by experience actions (in the batch ERB)
        slice_indices = np.zeros([self.batch_buffer_size, 2])
        slice_indices[:, 0] = np.arange(self.batch_buffer_size)
        slice_indices[:, 1] = action_append

        # Pre-computed V-values (used in tape_1)
        next_state_value = self.tar_state_value(next_state_distance_batch_tensor)
        # Use np.log10 because the input argument is not a tensor.
        current_policy_batch = self.actor(current_state_distance_batch_tensor)
        current_policy_action_batch = np.random.choice(self.actions,
                                                       size=self.batch_buffer_size,
                                                       p=current_policy_batch.numpy()[0])

        v_slice_indices = np.zeros([self.batch_buffer_size, 2])
        v_slice_indices[:, 0] = np.arange(self.batch_buffer_size)
        v_slice_indices[:, 1] = current_policy_action_batch

        log_prob_selected_action_batch = tf.gather_nd(current_policy_batch, indices=v_slice_indices.astype(int))
        log_prob_selected_action_batch = tf.divide(tf.math.log(log_prob_selected_action_batch), 2.303)
        log_prob_selected_action_batch_tensor = tf.convert_to_tensor(log_prob_selected_action_batch, dtype=tf.float32)

        # Pre-computed state-action value
        state_action_value_pred = self.action_value(current_state_distance_batch_tensor)
        state_action_value_pred_slice = tf.gather_nd(state_action_value_pred, indices=v_slice_indices.astype(int))
        epsilon = np.zeros([len(self.actions)], dtype=float)

        # # Re-parameterised action used for updating the policy network
        # # Determine the epsilon
        # # If S_{t+1} is terminated --> It means there is obstacle in front of it --> epsilon = [0.05, 0.45, 0.45, 0.05]
        # # If S_{t+1} is not terminated,
        # # --> It means there is no obstacle in front of it --> epsilon = [0.75, 0.1, 0.1, 0.05]
        # for k in range(self.batch_buffer_size):
        #     if self.death_flag_data[indices[k]]:
        #         # S_{t+1} is terminated --> Encourage to go left or right.
        #         epsilon_t = [0.00, 0.50, 0.50, 0.00]
        #     else:
        #         # S_{t+1} is not terminated --> Encourage to go forward.
        #         epsilon_t = [0.8, 0.1, 0.1, 0]
        #     re_param_action[k] = np.random.choice(self.actions, p=epsilon_t)

        # epsilon_t = [0.5, 0.25, 0.25, 0.0]
        for i in range(len(epsilon)):
            if i == 0:
                epsilon[i] = np.random.random(1)
            elif i == len(epsilon) - 1:
                epsilon[i] = 1 - np.sum(epsilon)
            else:
                epsilon[i] = (1 - np.sum(epsilon)) * np.random.random(1)
        # np.random.shuffle(epsilon)
        re_param_action = np.random.choice(self.actions, size=self.batch_buffer_size, p=epsilon)

        # # Slice \pi by re-param actions
        pi_slice_indices = np.zeros([self.batch_buffer_size, 2])
        pi_slice_indices[:, 0] = np.arange(self.batch_buffer_size)
        pi_slice_indices[:, 1] = re_param_action

        # ********************************* Update the models for 'self.epoch' epochs **********************************
        for m in range(self.epoch):
            # ----------------------------- Update the State-Action Value network (Q) start ----------------------------
            with tf.GradientTape() as tape_1:
                # Tape_1 --> Update Q network
                tape_1.watch(current_state_distance_batch_tensor)
                tape_1.watch(next_state_value)
                tape_1.watch(reward_batch_tensor)
                state_action_value = self.action_value(current_state_distance_batch_tensor)
                state_action_value_action = gather_nd(state_action_value, indices=slice_indices.astype(int))
                state_action_value_2t = tf.math.add(reward_batch_tensor, tf.math.multiply(gamma, next_state_value))
                # State_Action_Value_Cost (L_Q)
                state_action_value_cost = tf.math.reduce_mean(
                    tf.math.divide(
                        tf.math.square(
                            tf.math.subtract(state_action_value_action, state_action_value_2t)), 2))
            # Compute for the gradient of state_action_value network (Q) (Tape_1)
            state_action_value_cost_grad = tape_1.gradient(state_action_value_cost, self.action_value.trainable_variables)
            self.action_value.optimizer.apply_gradients(zip(state_action_value_cost_grad, self.action_value.trainable_variables))
            action_value_glob_gradient = tf.linalg.global_norm(state_action_value_cost_grad)  # Compute global norm grad
            self.action_value_grad_glob_norm = action_value_glob_gradient.numpy()  # Store the np value.
            self.action_value.save('C:/MHOO_2/Master/ACS6300_Project/Airsim/SAC/Trained_Models/SAC_01/trained_SAC_Q_01.h5')
            # ------------------------------ Update the State-Action Value network (Q) end -----------------------------

            # --------------------------------- Update the State Value network (V) start -------------------------------
            with tf.GradientTape() as tape_2:
                # Tape_2
                tape_2.watch(current_state_distance_batch_tensor)
                tape_2.watch(state_action_value_pred_slice)
                tape_2.watch(log_prob_selected_action_batch_tensor)
                state_value = self.state_value(current_state_distance_batch_tensor)
                state_value_2t = tf.math.subtract(state_action_value_pred_slice, log_prob_selected_action_batch_tensor)
                state_value_2t = tf.math.reduce_mean(state_value_2t)
                # State_Value_Cost (L_V)
                state_value_cost = tf.math.reduce_mean(
                    tf.math.divide(
                        tf.math.square(
                            tf.math.subtract(state_value, state_value_2t)), 2))
            # Compute for the gradient of state-value network (V) (Tape_2)
            state_value_cost_grad = tape_2.gradient(state_value_cost, self.state_value.trainable_variables)
            self.state_value.optimizer.apply_gradients(zip(state_value_cost_grad, self.state_value.trainable_variables))
            state_value_glob_gradient = tf.linalg.global_norm(state_value_cost_grad)  # Compute global norm grad
            self.state_value_grad_glob_norm = state_value_glob_gradient.numpy()  # Store the np value.
            self.state_value.save('C:/MHOO_2/Master/ACS6300_Project/Airsim/SAC/Trained_Models/SAC_01/trained_SAC_V_01.h5')
            # ---------------------------------- Update the State Value network (V) end --------------------------------

            # ----------------------------------- Update the Policy network (\pi) start --------------------------------
            with tf.GradientTape() as tape_3:
                # Tape_3
                tape_3.watch(current_state_distance_batch_tensor)
                tape_3.watch(state_action_value_pred)
                policy_pred = self.actor(current_state_distance_batch_tensor)
                re_param_policy_append = gather_nd(policy_pred, indices=pi_slice_indices.astype(int))
                log_re_param_policy_append = tf.math.divide(tf.math.log(re_param_policy_append), 2.303)
                state_action_value_3 = gather_nd(state_action_value_pred, indices=pi_slice_indices.astype(int))
                # Policy_Cost (L_\pi)
                policy_cost = -tf.math.reduce_mean(tf.math.subtract(log_re_param_policy_append, state_action_value_3))
            # Compute for the gradient of policy network (\pi) (Tape_3)
            policy_cost_grad = tape_3.gradient(policy_cost, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(policy_cost_grad, self.actor.trainable_variables))
            actor_glob_gradient = tf.linalg.global_norm(policy_cost_grad)  # Compute global norm grad
            self.actor_grad_glob_norm = actor_glob_gradient.numpy()  # Store the np value.
            self.actor.save('C:/MHOO_2/Master/ACS6300_Project/Airsim/SAC/Trained_Models/SAC_01/trained_SAC_PI_01.h5')
        # ------------------------------------ Update the Policy network (\pi) end -------------------------------------

        # Update the Target State Value Networks (V') (Soft Updating)
        self.update_count += 1
        if self.update_count >= self.update_period:
            tau = 0.005  # Soft updating rate
            # Update the Target State Value Network (V_bar)
            psi = self.state_value.weights
            psi_bar = self.tar_state_value.weights
            for m in range(len(psi)):
                psi_bar[m] = (tau * psi[m]) + ((1 - tau) * psi_bar[m])
            self.tar_state_value.set_weights(psi_bar)
            # Reset updating flag
            self.update_count = 0
    # -------------------------------------------- Networks Training End -----------------------------------------------

    # ---------------------------------------------- Main loop start ---------------------------------------------------
    def agent_training(self, episode):
        for i in range(episode):
            self.drone.reset()  # Reset the drone
            self.drone.enableApiControl(True)  # Enable API control for airsim
            pose = self.drone.simGetVehiclePose()
            init_y = randrange(0, 100)
            pose.position.y_val = init_y
            self.drone.simSetVehiclePose(pose, False)
            self.takeoff(0)  # Take off the drone

            # Reset death_flag and state
            self.death_flag = False  # Death flag
            self.terminated = False  # Terminated flag
            fit = False  # Fit flag
            sum_episode_reward = 0  # Cumulative reward for each episode
            step_count = 0

            # Get the initial location of the agent
            self.drone_coordinates = self.coordinates()
            self.final_coordinates = [100, 0]
            self.final_coordinates[1] = init_y
            distance_target = np.sqrt((self.final_coordinates[0] - self.drone_coordinates[0])**2 +
                                      (self.final_coordinates[1] - self.drone_coordinates[1])**2)
            self.max_dis = distance_target
            self.prev_dis = distance_target
            # The drone progression (in percent)
            self.distance_target = 100 - (100 * distance_target / self.max_dis)

            # ----------------------------------- Prepare current state input (S_t) start ------------------------------
            # Current state 1 (S_t)
            # Get the distance sensors data
            self.state = [self.drone.getDistanceSensorData("DistanceWest").distance,
                          self.drone.getDistanceSensorData("DistanceNorthWest").distance,
                          self.drone.getDistanceSensorData("DistanceNorth").distance,
                          self.drone.getDistanceSensorData("DistanceNorthEast").distance,
                          self.drone.getDistanceSensorData("DistanceEast").distance,
                          self.drone.getDistanceSensorData("DistanceSouthEast").distance,
                          self.drone.getDistanceSensorData("DistanceSouth").distance,
                          self.drone.getDistanceSensorData("DistanceSouthWest").distance]
            # ------------------------------------ Prepare current state input (S_t) end -------------------------------

            # ------------------- Loop for the operations (Time step = self.batch_buffer_size) -------------------------
            while not self.terminated:
                # Choose action from the policy network (Stochastic Policy)
                state_sens = np.expand_dims(self.state, axis=0)
                state_sens_tensor = tf.convert_to_tensor(state_sens)
                policy = self.actor(state_sens_tensor)
                selected_action = np.random.choice(self.actions, p=policy.numpy()[0])

                # selected_action = np.unravel_index(np.argmax(policy[0]), policy.shape)
                # selected_action = selected_action[1]
                # Get the policy of the selected action
                p_action = gather_nd(policy, indices=(0, selected_action))
                # print('policy: ', policy, ', selected_action: ', selected_action)

                # Environment <-- Reward shaping
                self.reward = self.environment(selected_action)
                sum_episode_reward = sum_episode_reward + self.reward

                # Store the tuples in the Experience Replay Buffer
                if self.step > (self.buffer_size - 1):
                    self.step = 0
                    self.full_flag = True

                # Store current images (S_t) and next images (S_{t+1})
                self.current_state_sens[self.step] = self.state
                self.p_action_data[self.step] = p_action
                self.action_data[self.step] = selected_action
                self.death_flag_data[self.step] = self.death_flag
                self.reward_data[self.step] = self.reward
                self.next_state_sens[self.step] = self.next_state

                # Step increment
                self.step += 1
                step_count += 1

                # Progression
                distance_target = np.sqrt((self.final_coordinates[0] - self.next_drone_coordinates[0])**2 +
                                          (self.final_coordinates[1] - self.next_drone_coordinates[1])**2)
                self.progress = 100 - (100 * distance_target / self.max_dis)

                # Update the state
                if self.death_flag or step_count >= self.fit_step:
                    # If the next state is death, terminated.
                    self.terminated = True
                    if self.step >= self.batch_buffer_size or self.full_flag:
                        fit = True
                else:
                    # If the next state is not death, continue to the next step.
                    self.state = self.next_state  # Get the next image as current img
                    self.distance_target = self.next_distance_target  # Get the next distance target as current
                    self.drone_coordinates = self.next_drone_coordinates  # Get the next location as current location

            # If the fit flag is true --> Train The Network
            if fit:
                if not self.full_flag:
                    # If the ERB is not full, only sampling from the size within the step count number.
                    # Shuffle the indices of buffer
                    random_indices = np.arange(self.step)
                    np.random.shuffle(random_indices)
                    # Update the action-value network after the episode is terminated.
                    self.network_training(random_indices)
                else:
                    # If the ERB is full already, sampling from the whole size ERB.
                    # Shuffle the indices of buffer
                    random_indices = np.arange(self.buffer_size)
                    np.random.shuffle(random_indices)
                    # Update the action-value network after the episode is terminated.
                    self.network_training(random_indices)

            # Store Episode's Reward
            self.append_reward.append(sum_episode_reward)
            np.save('C:/MHOO_2/Master/ACS6300_Project/Airsim/SAC/Append_Reward/SAC_01/append_reward_SAC_01', self.append_reward)

            if (self.training_record + i) % 100 == 0:
                self.actor.save(f'C:/MHOO_2/Master/ACS6300_Project/Airsim/SAC/Trained_Models/SAC_01/trained_SAC_{self.training_record+i}.h5')
                np.savez('C:/MHOO_2/Master/ACS6300_Project/Airsim/SAC/ERB/SAC_01/ERB_SAC_01',
                         self.current_state_sens, self.p_action_data, self.action_data,
                         self.death_flag_data, self.reward_data, self.next_state_sens)

            # Print the status of Learning.
            print('Episode: ', i, ', Step: ', self.step, ', Sum_reward: ', sum_episode_reward, ', Avg_reward: ',
                  np.sum(self.append_reward)/len(self.append_reward), ', Progression: ', self.progress,
                  ', Actor_grad: ', self.actor_grad_glob_norm, ', State_grad: ', self.state_value_grad_glob_norm,
                  ', Action_grad: ', self.action_value_grad_glob_norm)
    # ---------------------------------------------- Main loop end -----------------------------------------------------
# ****************************************** Soft Actor-Critic class end ***********************************************


# ******************************************* Main Program Start *******************************************************
# Training the drone
if __name__ == "__main__":
    droneSAC = SACLearning()  # <-- Create drone Soft-Actor-Critic object
    droneSAC.agent_training(5000)  # <-- Training the drone (episode)
# ******************************************* Main Program End *********************************************************
