# Neural network as a function approximator to the q-matrix, via DQN algorithm. Experience replay addition.
# We have increased the number of time steps we fed into the DQN as well as putting in an extra feature extraction step,
# which has now proven to shown to converge to ALL the trained as well as untrained on modes of failures.

# NOTE: Batch number 48 is key.

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras

class DQN_varying_time_steps:
    """
    This class implements the DQN algorithm to solve a simple environment model which has the state space equation:

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


    def __init__(self, x_limit=10, u_limit = 3, given_time_steps=1, epsilon=1, 
                 possible_b_vector=[1, -1], possible_a_vector=[1.1, -1.1], 
                 number_of_episodes_per_batch=100, number_of_batches=5000,
                 unseen_a_vector=[1, -1]):
                 
                 self.x_limit = x_limit
                 self.u_limit = u_limit
                 self.epsilon = epsilon
                 self.possible_b_vector = possible_b_vector
                 self.possible_a_vector = possible_a_vector
                 self.unseen_a_vector = unseen_a_vector
                 self.number_of_episodes_per_batch = number_of_episodes_per_batch
                 self.number_of_batches = number_of_batches

                 # Initialize cost per batch vector
                 self.cost_per_batch = []

                 # NOTE: The equations for initializing state_size, time steps, etc. can be found under implementation notes in section [3] in the 
                 # 'Lent Term Work' document.
                 self.given_time_steps = given_time_steps
                 self.state_size = given_time_steps * 2 + 1
                 self.time_steps = given_time_steps + 9
                 self.starting_time_step_for_RL_trajectory = given_time_steps

                 # Parameters required for creating a NN
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

        # Simple sequential neural network
        model = keras.Sequential()
        init = tf.keras.initializers.HeUniform()
        
        # Automatic feature extraction (via CNN for example, can explore different architectures as well)
        model.add(keras.layers.Reshape((11, 1)))
        model.add(keras.layers.Conv1D(128, kernel_size=5, activation='relu', strides=1))
        model.add(keras.layers.Conv1D(128, kernel_size=5, activation='relu', strides=1))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        
        # DQN algorithm
        model.add(keras.layers.Dense(24, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(24, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer=init)) 
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

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
            for k in range(self.starting_time_step_for_RL_trajectory, self.time_steps):  # going through the time steps from k=1 onwards (as we need to initialize at k=0)
                
                # NOTE: Refactoring code such that we can change no. given time steps fed into NN without having to copy paste code.
                # Creating the current_state array that needs to be fed into the NN, but in an extendable way.
                temp_array = [0] * self.state_size
                temp_array[0] = self.x[k]
                for i in range(self.given_time_steps):
                    temp_array[i*2 + 1] = self.x[k - (i + 1)]
                    temp_array[i*2 + 2] = self.u[k - (i + 1)]
                
                # Get current state
                current_state = np.array(temp_array)
                current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size

                # Get action (index as well as actual action done)
                selected_action = self.choose_action(current_state)   # choose best action based on 'current' augmented state, via epsilon greedy algorithm.
                self.u[k] = selected_action - self.u_limit      # convert argmax of output of NN to the actual action we are taking, with index=0 of NN representing u=-10.

                # Basically limits x to x_limit and -x_limit for next state, and updates next state
                self.x[k+1] = self.a * self.x[k] + self.b * self.u[k]

                # Calculate reward
                reward = - (self.x[k+1]**2)

                # NOTE: Refactored code to make it easily extendable
                # Get next augmented state
                temp_array_2 = [0] * self.state_size
                temp_array_2[0] = self.x[k + 1]
                for i in range(self.given_time_steps):
                    temp_array_2[i*2 + 1] = self.x[k - (i + 1) + 1]
                    temp_array_2[i*2 + 2] = self.u[k - (i + 1) + 1]
                next_state = np.array(temp_array_2)
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
            # NOTE: refactored code so that it is easily extendable
            self.x[0] = random.randint(-3, 3)
            self.u[0] = random.randint(-3, 3)

            for i in range(1, self.given_time_steps):
                self.u[i] = random.randint(-3, 3)
                
            for j in range(self.given_time_steps):
                self.x[j+1] = self.a * self.x[j] + self.b * self.u[j]

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
                # NOTE: refactored code so that it is easily extendable
                self.x[0] = 1
                self.u[0] = 1

                for i in range(1, self.given_time_steps):
                    self.u[i] = 1
                    
                for j in range(self.given_time_steps):
                    x_values[j+1] = a * x_values[j] + b * u_values[j]

                # Running test trajectories using RL agent
                for k in range(self.starting_time_step_for_RL_trajectory, self.time_steps):
                    # NOTE: Refactoring code such that we can change no. given time steps fed into NN without having to copy paste code.
                    # Creating the current_state array that needs to be fed into the NN, but in an extendable way.
                    temp_array = [0] * self.state_size
                    temp_array[0] = x_values[k]
                    for q in range(self.given_time_steps):
                        temp_array[q*2 + 1] = x_values[k - (q + 1)]
                        temp_array[q*2 + 2] = u_values[k - (q + 1)]
                
                    # Choose max q-value action, minimised over all the possible actions (u(k))
                    current_state = np.array(temp_array)
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
                    # NOTE: refactored code so that it is easily extendable
                    x_values[0] = 1
                    u_values[0] = 1

                    for i in range(1, self.given_time_steps):
                        u_values[i] = 1
                        
                    for j in range(self.given_time_steps):
                        x_values[j+1] = a * x_values[j] + b * u_values[j]

                    # Running test trajectories using RL agent
                    for k in range(self.starting_time_step_for_RL_trajectory, self.time_steps):
                        # NOTE: Refactoring code such that we can change no. given time steps fed into NN without having to copy paste code.
                        # Creating the current_state array that needs to be fed into the NN, but in an extendable way.
                        temp_array = [0] * self.state_size
                        temp_array[0] = x_values[k]
                        for q in range(self.given_time_steps):
                            temp_array[q*2 + 1] = x_values[k - (q + 1)]
                            temp_array[q*2 + 2] = u_values[k - (q + 1)]
                    
                        # Choose max q-value action, minimised over all the possible actions (u(k))
                        current_state = np.array(temp_array)
                        current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size

                        current_q_values = self.model.predict(current_state)[0]   # grab the q-values over all possible actions in the current state, outputted by NN.
                        max_q_index = np.argmax(current_q_values)
                        u_values[k] = max_q_index - self.u_limit # Does action corresponding to minimum cost

                        # Basically limits x to x_limit and -x_limit for next state, and updates next state
                        x_values[k+1] = a * x_values[k] + b * u_values[k]

                        # Calculates and stores cost at each time step for the particular 'a' and 'b' combination
                        self.cost[combination_index][k] = np.clip(x_values[k+1], -self.x_limit, self.x_limit)**2

                    # Plots either trajectory or error over time steps
                    if option == "trajectory":
                        plt.plot(range(self.time_steps + 1), np.clip(x_values, -self.x_limit, self.x_limit), label=f'a = {a}, b = {b}')  # plot the given trajectory for a single combination of 'a' and 'b' value
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
    agent = DQN_varying_time_steps(given_time_steps=5, number_of_episodes_per_batch=10, number_of_batches=100000)  # (1) X = number of batches until convergence
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