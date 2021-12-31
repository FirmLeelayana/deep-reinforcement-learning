# Neural network as a function approximator to the q-matrix, via DQN algorithm.

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras

class DQN:
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


    def __init__(self, x_limit=10, u_limit = 10, time_steps=10, epsilon=1, 
                 possible_b_vector=[1,-1], possible_a_vector=[2,-2], 
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
                 self.state_size = 3    # Augmented state = xk xk-1 uk-1, 3 parameters
                 self.action_size = u_limit * 2 + 1     # Action size = the values 'u' can take
                 self.memory = deque(maxlen=2000)   # Experience replay buffer, to sample from
                 self.gamma = 0.95    # discount rate
                 self.epsilon = 1.0  # exploration rate (initial)
                 self.epsilon_max = 1.0     # maximum epsilon value
                 self.epsilon_min = 0.01    # minimum epsilon value
                 self.epsilon_decay = 0.01     # decay for our epsilon initial value as we run episodes
                 self.learning_rate = 0.001     # learning rate, alpha
                 self.model = self._build_model()   # create our neural network model


    def _build_model(self):
        """Creates our neural network model architecture."""

        model = keras.Sequential()  # simple sequential neural network, with 3 fully/densely connected layers
        init = tf.keras.initializers.HeUniform()    # specify initializer for weights
        model.add(keras.layers.Dense(24, input_shape=(self.state_size,), activation='relu', kernel_initializer=init))    # Hidden Layer 1 = 24 nodes, ReLU activation function, He init.
        model.add(keras.layers.Dense(24, activation='relu', kernel_initializer=init))   # Hidden Layer 2 = 24 nodes, ReLU activation function, He init.
        model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer=init))   # Output Layer = (action_size) nodes, linear activation function, He init.
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))    # Loss = Mean Squared Error Loss, Adam Optimizer with learning rate alpha.

        return model

    
    def reset_agent(self):
        """Resets the neural network to its default, untrained state."""

        self.model = self._build_model()   # reset NN model

    
    def choose_action(self, state):
        """Method to get our next action."""

        # Selects random action with prob=epsilon, else action=maxQ (epsilon greedy policy)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)   # pick random action

        act_values = self.model.predict(state)  # NN makes prediction of Q values for current state
        return np.argmax(act_values[0])     # pick action that maximizes Q value


    def run_one_episode_and_train(self):
        """
        Trains the agent over a single episode.
        """

        # Exploration step - either following the optimal policy, or exploring randomly
        for k in range(1, self.time_steps):  # going through the time steps from k=2 onwards (as we need to initialize at k=0)
            current_state = np.array([self.x[k], self.x[k-1], self.u[k-1]])
            current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size
            selected_action = self.choose_action(current_state)   # choose best action based on 'current' augmented state, via epsilon greedy algorithm.
            self.u[k] = selected_action - self.u_limit      # convert argmax of output of NN to the actual action we are taking, with index=0 of NN representing u=-10.

            # Basically limits x to x_limit and -x_limit for next state, and updates next state
            self.x[k+1] = min(max(self.a * self.x[k] + self.b * self.u[k], -self.x_limit), self.x_limit)

        # Learning step (epsilon greedy, off-policy)
        for k in range(1, self.time_steps):

            # Calculate reward
            reward = - (self.x[k]**2 + self.u[k]**2)    # reward is negative cost

            # Get current augmented state, and associated q-value from output of NN
            current_state = np.array([self.x[k], self.x[k-1], self.u[k-1]])   # get current augmented state
            current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size
            current_q_values = self.model.predict(current_state)    # grab the q-values over all possible actions in the current state, outputted by NN.
            chosen_action_index = int(self.u[k] + self.u_limit)      # convert chosen action to the index corresponding to that action for the output of the NN.
            
            # Get next state q-value via output of NN, and corresponding td_target
            next_state = np.array([self.x[k+1], self.x[k], self.u[k]])   # get next augmented state
            next_state = next_state.reshape(-1, self.state_size)  # reshape to correct size
            next_q_values = self.model.predict(next_state)[0]    # grab the q-values over all possible actions in the next state, outputted by NN.
            td_target = (reward + self.gamma * np.amax(next_q_values))     # takes the maximum q-value over all possible actions at the next state -> Temporal difference target
            
            # Update outputted q_values (given by NN) at chosen action index with the td_target (Bellman equation) -> i.e create the 'true' labels output vector
            current_q_values[0][chosen_action_index] = td_target    #(e.g. alpha = 1)

            # Train the network -> give it the input to NN (augmented state), and then give it the 'true' label of what output of NN should be, which is current_q_values, which 
            # has been updated at the 'chosen_action_index' index with the Bellman equation update. (i.e. telling model it should train NN weights such that output should now give the 
            # Bellman-equation-updated Q-values instead).
            self.model.fit(current_state, current_q_values, epochs=1, verbose=0)    # train the model


    def run_one_batch_and_train(self):
        """Runs a single batch, comprising of a number of episodes, training the agent."""

        self.b = random.choice(self.possible_b_vector)  # Randomly selects the B value (failure mode)
        self.a = random.choice(self.possible_a_vector)  # Randomly selects the A value

        self.u = np.zeros(self.time_steps)
        self.x = np.zeros(self.time_steps + 1)  # As we need to index x[k+1] for the last time step as well

        # Selects a number randomly between -x_limit and x_limit, and places it in x[0] and u[0].
        self.x[0] = random.randint(-self.x_limit, self.x_limit)
        self.u[0] = random.randint(-self.u_limit, self.u_limit)
        
        # Starts at 1 as you need the previous x and u values as history buffer, to make it a Markovian process.
        self.x[1] = min(max(self.a * self.x[0] + self.b * self.u[0], -self.x_limit), self.x_limit)

        for _ in range(self.number_of_episodes_per_batch):
            self.run_one_episode_and_train()

    
    def run_multiple_batches_and_train(self):
        """
        Trains agent over the specified number of batches, each batch consisting of multiple episodes.
        """

        self.cost_per_batch = []
        for i in range(self.number_of_batches):
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
        
        # Initialize x and u vectors over the time steps
        x_values = np.zeros(self.time_steps + 1)
        u_values = np.zeros(self.time_steps)

        # Iterating over all possible combinations of a and b values.
        for b in self.possible_b_vector:
            for a in a_vector:
                # Initializing x and u values.
                x_values[1] = self.x_limit / 5  # testing agent on a step impulse
                x_values[0] = 0
                u_values[0] = 0

                for k in range(1, self.time_steps):
                    # Choose max q-value action, minimised over all the possible actions (u(k))
                    current_state = np.array([x_values[k], x_values[k-1], u_values[k-1]])   # get current augmented state
                    current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size
                    current_q_values = self.model.predict(current_state)[0]    # grab the q-values over all possible actions in the current state, outputted by NN.
                    max_q_index = np.argmax(current_q_values)
                    u_values[k] = max_q_index - self.u_limit # Does action corresponding to minimum cost

                    # Calculates and stores cost at each time step for the particular 'a' and 'b' combination
                    cost_matrix[combination_index][k] = x_values[k]**2 + u_values[k]**2

                    # Stores state at each time step for the particular 'a' and 'b' combination
                    state_matrix[combination_index][k] = x_values[k]

                    # Basically limits x to x_limit and -x_limit for next state, and updates next state
                    x_values[k+1] = min(max(a * x_values[k] + b * u_values[k], -self.x_limit), self.x_limit)

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
        
        a_vector = self.possible_a_vector.copy()
        a_vector.extend(self.unseen_a_vector)
        a_vector.sort()

        # Initialize cost matrix for each possible combination at each time step
        total_number_combinations = len(a_vector) * len(self.possible_b_vector)
        self.cost = np.zeros((total_number_combinations, self.time_steps))
        combination_index = 0  # represents current combination index
        
        # Initialize x and u vectors over the time steps
        x_values = np.zeros(self.time_steps + 1)
        u_values = np.zeros(self.time_steps)

        plt.clf()  # clears the current figure

        # Iterating over ALL possible combinations of a and b values.
        for b in self.possible_b_vector:
            for a in a_vector:
                # Initializing x and u values.
                x_values[1] = self.x_limit / 5  # testing agent on a step impulse
                x_values[0] = 0
                u_values[0] = 0

                for k in range(1, self.time_steps):
                    # Choose max q-value action, minimised over all the possible actions (u(k))
                    current_state = np.array([x_values[k], x_values[k-1], u_values[k-1]])   # get current augmented state
                    current_state = current_state.reshape(-1, self.state_size)  # reshape to correct size
                    current_q_values = self.model.predict(current_state)[0]    # grab the q-values over all possible actions in the current state, outputted by NN.
                    max_q_index = np.argmax(current_q_values)
                    u_values[k] = max_q_index - self.u_limit # Does action corresponding to minimum cost

                    # Calculates and stores cost at each time step for the particular 'a' and 'b' combination
                    self.cost[combination_index][k] = x_values[k]**2 + u_values[k]**2

                    # Basically limits x to x_limit and -x_limit for next state, and updates next state
                    x_values[k+1] = min(max(a * x_values[k] + b * u_values[k], -self.x_limit), self.x_limit)

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
                print(f"Current batch number: {i}, average cost per trajectory of {self.cost.mean(axis=1)}, overall average cost: {self.cost.mean()}")


if __name__ == "__main__":
    # Will plot the trajectories/error graphs of the 4 different 'a' and 'b' combinations at every 
    # batch_number_until_plot batches, up to maximum of number_of_batches. Will also
    # print out the current batch number to terminal, as well as average cost per trajectory
    # and overall average cost.

    # Fix random seeds
    RANDOM_SEED = 1000
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Initialize the number of batches and episodes per batch variables (for training)
    agent = DQN(number_of_episodes_per_batch=10, number_of_batches=100000)  # (1) X = number of batches until convergence

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