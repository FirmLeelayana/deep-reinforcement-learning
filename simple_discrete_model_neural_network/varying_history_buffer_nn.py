# Neural network as a function approximator to the q-matrix, via DQN algorithm. Experience replay addition, with more history buffer (20 time steps in total in this case).
# NOTE: This is the old version of the file (for DQN with increasing history buffer for section 3), where this file (may have) implemented the increase in time
# steps wrongly in code. The more sure/updated version is located in varying_history_buffer_nn_ver2.py -> use this for the main findings.

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras

class DQN_history_buffer:
    """
    This class implements the DQN algorithm with more history buffer to solve a simple environment model which has the state space equation:

    x(k+1) = a * x(k) + b * u(k)

    The values of a and b are fixed per episode of a period of time steps, however it is changed randomly throughout the training. The failure mode
    of interest is the b variable, in which this would switch from -1 to 1 for example; this inversion represents the environment switching
    from a nominal mode to a failure mode. The ultimate aim of the q-learning algorithm is to learn an optimal q-table/policy such that
    it would be able to handle any variations in a or b.

    Initialization variables:
    x_limit = int, the limit as to which the state value can go up to. Default = 20.
    u_limit = int, the limit as to which the input value can go up to. Default = 20.
    time_steps = int, the number of time steps per episode. Default = 10.
    epsilon = int, the greedy epsilon value the q-learning will follow. Default = 1.
    possible_b_vector = array of ints, represents the possible gains that B could take. Default = [1, -1]
    possible_a_vector = array of ints, represents the possible values that A could take. Default = [1, 3]
    number_of_episodes_per_batch = int, number of episodes per training batch. Default = 100.
    number_of_batches = int, number of batches in total. Each batch will train the same Q table, but after a batch, the plot of how well the policy does
                        will be created. Default = 5000.
    unseen_a_vector = array of unseen failure modes in a.
    """


    def __init__(self, x_limit=10, u_limit = 3, time_steps=30, epsilon=1, 
                 possible_b_vector=[1, -1], possible_a_vector=[1.2, -1.2], 
                 number_of_episodes_per_batch=100, number_of_batches=5000,
                 unseen_a_vector=[1, -1]):
                 
                 self.x_limit = x_limit
                 self.u_limit = u_limit
                 self.time_steps = time_steps
                 self.epsilon = epsilon
                 self.possible_b_vector = possible_b_vector
                 self.possible_a_vector = possible_a_vector
                 self.unseen_a_vector = unseen_a_vector
                 self.number_of_episodes_per_batch = number_of_episodes_per_batch
                 self.number_of_batches = number_of_batches

                 # Initialize cost per batch vector
                 self.cost_per_batch = []

                 # Parameters required for creating a NN
                 self.state_size = 41    # Augmented state
                 self.action_size = u_limit * 2 + 1     # Action size = the values 'u' can take
                 self.memory = deque(maxlen=1000)   # Experience replay buffer, to sample from
                 self.gamma = 0.95    # discount rate
                 self.epsilon = 1.0  # exploration rate (initial)
                 self.epsilon_max = 1.0     # maximum epsilon value
                 self.epsilon_min = 0.01    # minimum epsilon value
                 self.epsilon_decay = 0.01     # decay for our epsilon initial value as we run episodes
                 self.learning_rate = 0.001     # learning rate, alpha
                 self.batch_size = 128     # batch size for sampling memory replay buffer
                 self.model = self._build_model()   # create our neural network model


    def _build_model(self):
        """Creates our neural network model architecture."""

        model = keras.Sequential()  # simple sequential neural network, with 3 fully/densely connected layers
        init = tf.keras.initializers.HeUniform()    # specify initializer for weights
        model.add(keras.layers.Dense(24, input_shape=(self.state_size,), activation='relu', kernel_initializer=init))    # Hidden Layer 1 = 24 nodes, ReLU activation function, He init.
        model.add(keras.layers.Dense(24, activation='relu', kernel_initializer=init))   # Hidden Layer 2 = 24 nodes, ReLU activation function, He init.
        model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer=init))   # Output Layer = (action_size) nodes, linear activation function, He init.
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))    # Loss = mean squared error, Adam Optimizer with learning rate alpha.

        return model

    
    def reset_agent(self):
        """Resets the neural network to its default, untrained state."""

        self.model = self._build_model()   # reset NN model

    
    def choose_action(self, state):
        """Method to get our next action (index)."""

        with tf.device('/device:GPU:0'):

            # Selects random action with prob=epsilon, else action=maxQ (epsilon greedy policy)
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)   # pick random action (0 to 20 in this case)

            # Prediction is done over a dataset, so we need to wrap the single 'state' prediction in another np.array, representing dataset consisting of 1 datapoint.
            act_values = self.model.predict(state)  # NN makes prediction of Q values for current state
            return np.argmax(act_values[0])     # pick action that maximizes Q value


    def run_one_episode_and_append_to_memory(self):
        """
        Runs one episode, and appends all the transitions to a memory replay buffer.
        """

        with tf.device('/device:GPU:0'):

            # Exploration step - either following the optimal policy, or exploring randomly
            for k in range(2, self.time_steps):  # going through the time steps from k=1 onwards (as we need to initialize at k=0)
                # Get current state
                current_state = np.array([self.x[k], self.x[k-1], self.u[k-1], self.x[k-2], self.u[k-2], self.x[k-3], self.u[k-3], self.x[k-4], self.u[k-4], self.x[k-5], self.u[k-5],
                self.x[k-6], self.u[k-6], self.x[k-7], self.u[k-7], self.x[k-8], self.u[k-8], self.x[k-9], self.u[k-9], self.x[k-10], self.u[k-10],
                self.x[k-11], self.u[k-11], self.x[k-12], self.u[k-12], self.x[k-13], self.u[k-13], self.x[k-14], self.u[k-14], self.x[k-15], self.u[k-15],
                self.x[k-16], self.u[k-16], self.x[k-17], self.u[k-17], self.x[k-18], self.u[k-18], self.x[k-19], self.u[k-19], self.x[k-20], self.u[k-20]])
                current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size

                # Get action (index as well as actual action done)
                selected_action = self.choose_action(current_state)   # choose best action based on 'current' augmented state, via epsilon greedy algorithm.
                self.u[k] = selected_action - self.u_limit      # convert argmax of output of NN to the actual action we are taking, with index=0 of NN representing u=-10.

                # Basically limits x to x_limit and -x_limit for next state, and updates next state
                self.x[k+1] = self.a * self.x[k] + self.b * self.u[k]

                # Calculate reward
                reward = - (self.x[k+1]**2)

                # Get next augmented state
                next_state = np.array([self.x[k+1], self.x[k], self.u[k], self.x[k-1], self.u[k-1], self.x[k-3], self.u[k-3], self.x[k-4], self.u[k-4], self.x[k-5], self.u[k-5],
                self.x[k-6], self.u[k-6], self.x[k-7], self.u[k-7], self.x[k-8], self.u[k-8], self.x[k-9], self.u[k-9], self.x[k-10], self.u[k-10],
                self.x[k-11], self.u[k-11], self.x[k-12], self.u[k-12], self.x[k-13], self.u[k-13], self.x[k-14], self.u[k-14], self.x[k-15], self.u[k-15],
                self.x[k-16], self.u[k-16], self.x[k-17], self.u[k-17], self.x[k-18], self.u[k-18], self.x[k-19], self.u[k-19], self.x[k-20], self.u[k-20]])
                next_state = next_state.reshape(-1, self.state_size)  # reshape to correct size

                done = False

                # Check if episode is done
                if (self.x[k+1] >= self.x_limit) or (k == self.time_steps - 1):
                    done = True

                # Append to experience replay buffer
                self.memory.append([current_state, selected_action, reward, next_state, done])  # where selected action = 0 to 20 (is an index)

                if done:
                    break


    def train(self):
        """Sample from memory experience buffer, and train the model."""

        # Check replay memory is big enough
        MIN_REPLAY_SIZE = self.batch_size
        if len(self.memory) < MIN_REPLAY_SIZE:
            return None

        # Sample from memory buffer, of batch size = self.batch_size
        mini_batch = random.sample(self.memory, self.batch_size)

        # For every transition in mini-batch, we fit the model.
        for observation, action_index, reward, new_observation, done in mini_batch:
            if not done:
                q_next = self.model.predict(new_observation)[0]
                td_target = reward + self.gamma * np.amax(q_next) # takes the maximum q-value over all possible actions at the next state -> Temporal difference target
            else:
                td_target = reward

            # Update outputted q_values (given by NN) at chosen action index with the td_target (Bellman equation) -> i.e create the 'true' labels output vector
            current_state_q_value = self.model.predict(observation)
            current_state_q_value[0][action_index] = td_target    # (e.g. alpha = 1)

            # Train the network -> give it the input to NN (augmented state), and then give it the 'true' label of what output of NN should be, which is current_q_values, which 
            # has been updated at the 'chosen_action_index' index with the Bellman equation update. (i.e. telling model it should train NN weights such that output should now give the 
            # Bellman-equation-updated Q-values instead).
            self.model.fit(observation, current_state_q_value, epochs=1, verbose=0)


    def run_one_batch_and_train(self):
        """Runs a single batch, comprising of a number of episodes, training the agent."""

        self.b = random.choice(self.possible_b_vector)  # Randomly selects the B value (failure mode)
        self.a = random.choice(self.possible_a_vector)  # Randomly selects the A value

        for _ in range(self.number_of_episodes_per_batch):
            self.u = np.zeros(self.time_steps)
            self.x = np.zeros(self.time_steps + 1)  # As we need to index x[k+1] for the last time step as well

            # Selects a number randomly between -x_limit and x_limit, and places it in x[0] and u[0].
            # Fixed the initial x[0] u[0] value to be between a smaller range, as to not immediately threshold on the limits.
            self.x[0] = random.randint(-1, 1)
            self.u[0] = random.randint(-1, 1)
            self.u[1] = random.randint(-1, 1)
            self.u[2] = random.randint(-1, 1)
            self.u[3] = random.randint(-1, 1)
            self.u[4] = random.randint(-1, 1)
            self.u[5] = random.randint(-1, 1)
            self.u[6] = random.randint(-1, 1)
            self.u[7] = random.randint(-1, 1)
            self.u[8] = random.randint(-1, 1)
            self.u[9] = random.randint(-1, 1)
            self.u[10] = random.randint(-1, 1)
            self.u[11] = random.randint(-1, 1)
            self.u[12] = random.randint(-1, 1)
            self.u[13] = random.randint(-1, 1)
            self.u[14] = random.randint(-1, 1)
            self.u[15] = random.randint(-1, 1)
            self.u[16] = random.randint(-1, 1)
            self.u[17] = random.randint(-1, 1)
            self.u[18] = random.randint(-1, 1)
            self.u[19] = random.randint(-1, 1)
            
            # Starts at 1 as you need the previous x and u values as history buffer, to make it a Markovian process.
            self.x[1] = self.a * self.x[0] + self.b * self.u[0]
            self.x[2] = self.a * self.x[1] + self.b * self.u[1]
            self.x[3] = self.a * self.x[2] + self.b * self.u[2]
            self.x[4] = self.a * self.x[3] + self.b * self.u[3]
            self.x[5] = self.a * self.x[4] + self.b * self.u[4]
            self.x[6] = self.a * self.x[5] + self.b * self.u[5]
            self.x[7] = self.a * self.x[6] + self.b * self.u[6]
            self.x[8] = self.a * self.x[7] + self.b * self.u[7]
            self.x[9] = self.a * self.x[8] + self.b * self.u[8]
            self.x[10] = self.a * self.x[9] + self.b * self.u[9]
            self.x[11] = self.a * self.x[10] + self.b * self.u[10]
            self.x[12] = self.a * self.x[11] + self.b * self.u[11]
            self.x[13] = self.a * self.x[12] + self.b * self.u[12]
            self.x[14] = self.a * self.x[13] + self.b * self.u[13]
            self.x[15] = self.a * self.x[14] + self.b * self.u[14]
            self.x[16] = self.a * self.x[15] + self.b * self.u[15]
            self.x[17] = self.a * self.x[16] + self.b * self.u[16]
            self.x[18] = self.a * self.x[17] + self.b * self.u[17]
            self.x[19] = self.a * self.x[18] + self.b * self.u[18]
            self.x[20] = self.a * self.x[19] + self.b * self.u[19]

            self.run_one_episode_and_append_to_memory() # append to experience replay
        
        self.train()    # trains once per batch

    
    def run_multiple_batches_and_train(self):
        """
        Trains agent over the specified number of batches, each batch consisting of multiple episodes.
        """

        self.cost_per_batch = []
        for i in range(self.number_of_batches): # Epsilon decays corresponding to number of batches -> at 400 batches decays to min value.
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * i)
            self.run_one_batch_and_train()


    def simulate_single_test_epsiode(self, test_type='overall'):
        """
        Runs a single test episode, under the current trained policy (the current q table).

        Input:
        1. test_type, a string -> corresponds to which combinations of a and b values we want to test the agent on.
                                  overall = all combinations
                                  unseen = only unseen combinations
                                  seen = only on trained-on combinations of failure modes
        Returns:
        1. A cost matrix, of size (number_of_combinations, self.time_steps), representing cost at each time step,
           for each trajectory/sample, where each sample has a unique 'a' and 'b' value combination.
        2. A state matrix, of size (number_of_combinations, self.time_steps), representing state at each time step,
           for each trajectory/sample, where each sample has a unique 'a' and 'b' value combination.
        To be used for creating evaluation metrics for the agent.
        """

        # Initializes the specified possible combination of a and b values.
        if test_type == 'overall':
            a_vector = self.possible_a_vector.copy()
            a_vector.extend(self.unseen_a_vector)
            a_vector.sort()
        elif test_type == 'seen':
            a_vector = self.possible_a_vector
        elif test_type == 'unseen':
            a_vector = self.unseen_a_vector

        total_number_combinations = len(a_vector) * len(self.possible_b_vector)
        self.total_number_combinations_test_episode = total_number_combinations  # store total test combinations for use in evaluation
        
        # Initialize cost and state matrix for each possible combination at each time step.
        cost_matrix = np.zeros((total_number_combinations, self.time_steps))
        state_matrix = np.zeros((total_number_combinations, self.time_steps))
        combination_index = 0  # represents current combination index
        self.possible_combinations = {}  # initialize (a, b) combination matrix

        # Iterating over all possible combinations of a and b values.
        for b in self.possible_b_vector:
            for a in a_vector:
                # Initialize x and u vectors over the time steps
                x_values = np.zeros(self.time_steps + 1)
                u_values = np.zeros(self.time_steps)

                # Initializing x and u values.
                x_values[0] = 1
                u_values[0] = 1
                u_values[1] = 1
                x_values[1] = a * x_values[0] + b * u_values[0]
                x_values[2] = a * x_values[1] + b * u_values[1]

                for k in range(2, self.time_steps):
                    # Choose max q-value action, minimised over all the possible actions (u(k))
                    current_state = np.array([x_values[k], x_values[k-1], u_values[k-1], x_values[k-2], u_values[k-2]])
                    current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size
                    current_q_values = self.model.predict(current_state)[0]   # grab the q-values over all possible actions in the current state, outputted by NN.
                    max_q_index = np.argmax(current_q_values)
                    u_values[k] = max_q_index - self.u_limit # Does action corresponding to minimum cost

                    # Basically limits x to x_limit and -x_limit for next state, and updates next state
                    x_values[k+1] = a * x_values[k] + b * u_values[k]

                    # Calculates and stores cost at each time step for the particular 'a' and 'b' combination
                    cost_matrix[combination_index][k] = x_values[k+1]**2

                    # Stores state at each time step for the particular 'a' and 'b' combination
                    state_matrix[combination_index][k] = x_values[k]

                # Store (a, b) combination
                self.possible_combinations[combination_index] = (a, b)

                # Increment combination index by 1
                combination_index += 1

        return cost_matrix, state_matrix


    def run_multiple_batches_and_train_record_cost(self):
        """
        Trains agent over the specified number of batches, each batch consisting of multiple episodes, and record cost.
        """

        self.cost_per_batch = []
        for i in range(self.number_of_batches):
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * i)
            self.run_one_batch_and_train()

            # Record cost per batch
            self.cost_per_batch.append(np.mean(self.simulate_single_test_epsiode()[0]))


    def plot_test_episode(self, option='trajectory'):
        """
        Plots time step against the state value, for the current trained policy.
        Each single test episode will have it's own cost matrix, and it's own trajectory plot for ALL the different combinations (unseen + seen)

        option = string, represents which plot we want to see.
        """

        with tf.device('/device:GPU:0'):
        
            a_vector = self.possible_a_vector.copy()
            a_vector.extend(self.unseen_a_vector)
            a_vector.sort()

            # Initialize cost matrix for each possible combination at each time step
            total_number_combinations = len(a_vector) * len(self.possible_b_vector)
            self.cost = np.zeros((total_number_combinations, self.time_steps))
            combination_index = 0  # represents current combination index
            
            plt.clf()  # clears the current figure

            # Iterating over ALL possible combinations of a and b values.
            for b in self.possible_b_vector:
                for a in a_vector:
                    # Initialize x and u vectors over the time steps
                    x_values = np.zeros(self.time_steps + 1)
                    u_values = np.zeros(self.time_steps)

                    # Initializing x and u values.
                    x_values[0] = 1
                    u_values[0] = 1
                    u_values[1] = 1
                    u_values[2] = 1
                    u_values[3] = 1
                    u_values[4] = 1
                    u_values[5] = 1
                    u_values[6] = 1
                    u_values[7] = 1
                    u_values[8] = 1
                    u_values[9] = 1
                    u_values[10] = 1
                    u_values[11] = 1
                    u_values[12] = 1
                    u_values[13] = 1
                    u_values[14] = 1
                    u_values[15] = 1
                    u_values[16] = 1
                    u_values[17] = 1
                    u_values[18] = 1
                    u_values[19] = 1
                    x_values[1] = a * x_values[0] + b * u_values[0]
                    x_values[2] = a * x_values[1] + b * u_values[1]
                    x_values[3] = a * x_values[2] + b * u_values[2]
                    x_values[4] = a * x_values[3] + b * u_values[3]
                    x_values[5] = a * x_values[4] + b * u_values[4]
                    x_values[6] = a * x_values[5] + b * u_values[5]
                    x_values[7] = a * x_values[6] + b * u_values[6]
                    x_values[8] = a * x_values[7] + b * u_values[7]
                    x_values[9] = a * x_values[8] + b * u_values[8]
                    x_values[10] = a * x_values[9] + b * u_values[9]
                    x_values[11] = a * x_values[10] + b * u_values[10]
                    x_values[12] = a * x_values[11] + b * u_values[11]
                    x_values[13] = a * x_values[12] + b * u_values[12]
                    x_values[14] = a * x_values[13] + b * u_values[13]
                    x_values[15] = a * x_values[14] + b * u_values[14]
                    x_values[16] = a * x_values[15] + b * u_values[15]
                    x_values[17] = a * x_values[16] + b * u_values[16]
                    x_values[18] = a * x_values[17] + b * u_values[17]
                    x_values[19] = a * x_values[18] + b * u_values[18]
                    x_values[20] = a * x_values[19] + b * u_values[19]

                    for k in range(2, self.time_steps):
                        # Choose max q-value action, minimised over all the possible actions (u(k))
                        current_state = np.array([x_values[k], x_values[k-1], u_values[k-1], x_values[k-2], u_values[k-2], x_values[k-3], u_values[k-3], x_values[k-4], u_values[k-4], x_values[k-5], u_values[k-5],
                        x_values[k-6], u_values[k-6], x_values[k-7], u_values[k-7], x_values[k-8], u_values[k-8], x_values[k-9], u_values[k-9], x_values[k-10], u_values[k-10],
                        x_values[k-11], u_values[k-11], x_values[k-12], u_values[k-12], x_values[k-13], u_values[k-13], x_values[k-14], u_values[k-14], x_values[k-15], u_values[k-15],
                        x_values[k-16], u_values[k-16], x_values[k-17], u_values[k-17], x_values[k-18], u_values[k-18], x_values[k-19], u_values[k-19], x_values[k-20], u_values[k-20]])
                        current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size
                        current_q_values = self.model.predict(current_state)[0]   # grab the q-values over all possible actions in the current state, outputted by NN.
                        max_q_index = np.argmax(current_q_values)
                        u_values[k] = max_q_index - self.u_limit # Does action corresponding to minimum cost

                        # Basically limits x to x_limit and -x_limit for next state, and updates next state
                        x_values[k+1] = a * x_values[k] + b * u_values[k]

                        # Calculates and stores cost at each time step for the particular 'a' and 'b' combination
                        self.cost[combination_index][k] = x_values[k+1]**2

                    # Plots either trajectory or error over time steps
                    if option == "trajectory":
                        plt.plot(range(self.time_steps + 1), x_values, label=f'a = {a}, b = {b}')  # plot the given trajectory for a single combination of 'a' and 'b' value
                    elif option == "cost":
                        plt.plot(range(self.time_steps), self.cost[combination_index], label=f'a = {a}, b = {b}')  # plot the cost for the trajectory
                    
                    # Increment combination index by 1
                    combination_index += 1

            plt.legend(loc="upper left")  # add a legend
            plt.title(f"{option} plots of a single test episode")
            plt.xlabel("Time steps")
            plt.ylabel(f"{option}")
            plt.ion()  # turn on interactive mode
            plt.pause(0.01)  # allow time for GUI to load
            plt.show()
        

    def run_multiple_batches_and_plot(self, batch_number_until_plot=100, option='trajectory'):
        """
        Trains agent over the specified number of batches, each batch consisting of multiple episodes, and produces a plot
        showing either trajectory or cost of a single test episode, at specified (batch_number_until_plot) batch intervals.
        """

        for i in range(self.number_of_batches):
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * i)
            self.run_one_batch_and_train()

            # Create a plot for every x batches
            if i % batch_number_until_plot == 0:
                self.plot_test_episode(option)
                # Print batch number and cost vector (over all combinations)
                print(f"Current batch number: {i}, average cost per trajectory of {self.cost.mean(axis=1)}, overall average cost: {self.cost.mean()}, epsilon: {self.epsilon}")


if __name__ == "__main__":
    # Will plot the trajectories/error graphs of the 4 different 'a' and 'b' combinations at every 
    # batch_number_until_plot batches, up to maximum of number_of_batches. Will also
    # print out the current batch number to terminal, as well as average cost per trajectory
    # and overall average cost.

    # Fix random seeds
    RANDOM_SEED = 1000
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Initialize the number of batches and episodes per batch variables (for training)
    agent = DQN_history_buffer(number_of_episodes_per_batch=10, number_of_batches=100000)  # (1) X = number of batches until convergence
    # Total number of transitions per episode = self.time_steps = 9 ish
    # Total number of transitions per batch = number_of_episodes_per_batch * self.time_steps = 90 ish
    # Trains once per batch, with batch_size = 128
    # Repeat over X number of batches until convergence.

    # Option 1: Trains the agent, and plots the trajectory graph every batch_number_until_plot batches.
    # Basically shows the trajectory plot as it is training.
    agent.run_multiple_batches_and_plot(batch_number_until_plot=1, option = 'trajectory')
    plt.pause(100)  # Pause the final plot for 100 seconds
    agent.reset_agent()  # Reset agent

    # Option 2: Trains the agent, and plots the cost graph every batch_number_until_plot batches.
    # Basically shows the cost plot as it is training.
    agent.run_multiple_batches_and_plot(batch_number_until_plot=10, option = 'cost')
    plt.pause(5)  # Pause the final plot for 5 seconds
    agent.reset_agent()  # Reset agent