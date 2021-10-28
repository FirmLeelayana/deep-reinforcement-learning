# This file implements a simple environment model, with both discrete action and state spaces, where only a single mode of failure exists.
# This code has been ported from MATLAB into Python, with full credits to Prof. Glenn Vinnicombe for the initial implementation.

import numpy as np
import random
import matplotlib.pyplot as plt

class SimpleDiscreteQLearning:
    """
    This class implements the tabular epsilon greedy q-learning algorithm to solve a simple environment model which has the state space equation:

    x(k+1) = a * x(k) + b * u(k)

    The values of a and b are fixed per episode of a period of time steps, however it is changed randomly throughout the training. The failure mode
    of interest is the b variable, in which this would switch from -1 to 1 for example; this inversion represents the environment switching
    from a nominal mode to a failure mode. The ultimate aim of the q-learning algorithm is to learn an optimal q-table/policy such that
    it would be able to handle any variations in a or b.

    Inputs:
    x_limit = int, the limit as to which the state value can go up to. Default = 20.
    u_limit = int, the limit as to which the input value can go up to. Default = 20.
    time_steps = int, the number of time steps per episode. Default = 10.
    epsilon = int, the greedy epsilon value the q-learning will follow. Default = 1.
    possible_b_vector = array of ints, represents the possible gains that B could take. Default = [1, -1]
    possible_a_vector = array of ints, represents the possible values that A could take. Default = [1, 2]
    number_of_episodes_per_batch = int, number of episodes per training batch. Default = 100.
    number_of_batches = int, number of batches in total. Each batch will train the same Q table, but after a batch, the plot of how well the policy does
                        will be created. Default = 10,000.
    """


    def __init__(self, x_limit=20, u_limit = 20, time_steps=10, epsilon=1, 
                 possible_b_vector=[1,-1], possible_a_vector=[1,2], 
                 number_of_episodes_per_batch=100, number_of_batches=10000):
                 
                 self.x_limit = x_limit
                 self.u_limit = u_limit
                 self.time_steps = time_steps
                 self.epsilon = epsilon
                 self.possible_b_vector = possible_b_vector
                 self.possible_a_vector = possible_a_vector
                 self.number_of_episodes_per_batch = number_of_episodes_per_batch
                 self.number_of_batches = number_of_batches

                 # Creating a 4D matrix for the Q table. Dimensions for: x(k-1), u(k-1), x(k), against all possible u(k) values.
                 # The three initial dimensions represent the augmented state of the system required to make it Markovian.
                 # We initialize Q values to 100, with the Q matrix indicating cost. There are 2*x_limit + 1 overall possible
                 # discrete states per each state, as we are going from negative x_limit to positive x_limit range. Same applies
                 # to the number of discrete input states possible. 
                 self.q_table = np.full((2*x_limit + 1, 2*x_limit + 1, 2*u_limit + 1, 2*u_limit + 1), 100)
                 self.number_times_explored = np.full((2*x_limit + 1, 2*x_limit + 1, 2*u_limit + 1, 2*u_limit + 1), 0)
                 
                 # Setting cost to be 0 at optimal state.
                 self.q_table[x_limit, x_limit, u_limit, u_limit] = 0 


    def run_one_episode(self):
        """
        Runs a single episode in the q-learning algorithm.
        """

        # Exploration step - either following the optimal policy, or exploring randomly
        for k in range(1, self.time_steps):  # going through the time steps from k=2 onwards (as we need to initialize at k=0)
            if random.uniform(0, 1) < self.epsilon:  # exploring with probability epsilon (epsilon greedy)
                self.u[k] = random.randint(-self.u_limit, self.u_limit)  # u(k) will be updated with a random value between -u_limit and u_limit (explore)
            else:
                # Choose minimum cost action, minimised over all the possible actions (u(k))
                min_cost_index = np.argmin(self.q_table[int(self.x[k] + self.x_limit), int(self.x[k-1] + self.x_limit), int(self.u[k-1] + self.u_limit)])
                self.u[k] = min_cost_index - (self.u_limit + 1)  # Does action corresponding to minimum cost

            # Basically limits x to x_limit and -x_limit for next state, and updates next state
            self.x[k+1] = min(max(self.a * self.x[k] + self.b * self.u[k], -self.x_limit), self.x_limit)

        # Learning step (greedy, off-policy)
        for k in range(1, self.time_steps):
            # Grabs current count value for current augmented agent state and action, and increment by 1
            self.number_times_explored[int(self.x[k] + self.x_limit), int(self.x[k-1] + self.x_limit), int(self.u[k-1] + self.u_limit), int(self.u[k] + self.u_limit)] += 1
            count = self.number_times_explored[int(self.x[k] + self.x_limit), int(self.x[k-1] + self.x_limit), int(self.u[k-1] + self.u_limit), int(self.u[k] + self.u_limit)]

            # Normalization constant; the more the current state is explored, the less impact the new q values contribute (discount factor)
            norm_constant = (1/count)

            current_q_value = self.q_table[int(self.x[k] + self.x_limit), int(self.x[k-1] + self.x_limit), int(self.u[k-1] + self.u_limit), int(self.u[k] + self.u_limit)]
            cost = self.x[k]**2 + self.u[k]**2
            least_cost_action = min(self.q_table[int(self.x[k+1] + self.x_limit), int(self.x[k] + self.x_limit), int(self.u[k] + self.u_limit)])

            # Update Q table for the current augmented agent state (containing xk xk-1 uk-1) and current action uk
            self.q_table[int(self.x[k] + self.x_limit), int(self.x[k-1] + self.x_limit), 
                         int(self.u[k-1] + self.u_limit), int(self.u[k] + self.u_limit)] = (1-norm_constant) * current_q_value + norm_constant * (cost + least_cost_action)
    

    def run_one_batch(self):
        """Runs a single batch, comprising of a number of episodes."""

        self.b = random.choice(self.possible_b_vector)  # Randomly selects the B value
        self.a = random.choice(self.possible_a_vector)  # Randomly selects the A value (failure mode)

        self.u = np.zeros(self.time_steps)
        self.x = np.zeros(self.time_steps + 1)  # As we need to index x[k+1] for the last time step as well

        # Selects a number randomly between -x_limit and x_limit, and places it in x[1].
        # Starts at 1 as you need the previous x and u values as history buffer, to make it a Markovian process.
        self.x[1] = random.randint(-self.x_limit, self.x_limit)

        for i in range(self.number_of_episodes_per_batch):
            self.run_one_episode()

    
    def plot_test_episode(self, option='trajectory'):
        """
        Plots time step against the state value, for the current trained policy.
        Each single test episode will have it's own cost matrix, and it's own trajectory plot for all the different combinations.

        option = string, represents which plot we want to see.
        """

        # Initialize cost matrix for each possible combination at each time step
        total_number_combinations = len(self.possible_a_vector) + len(self.possible_b_vector)
        self.cost = np.zeros((total_number_combinations, self.time_steps))
        combination_index = 0  # represents current combination index
        
        # Initialize x and u vectors over the time steps
        x_values = np.zeros(self.time_steps + 1)
        u_values = np.zeros(self.time_steps)

        plt.clf()  # clears the current figure

        for b in self.possible_b_vector:
            for a in self.possible_a_vector:
                # Iterating over all possible combinations of a and b values.
                # Initializing x and u values.
                x_values[1] = self.x_limit / 5  # testing agent on a step impulse for example
                x_values[0] = 0
                u_values[0] = 0

                for k in range(1, self.time_steps):
                    # Choose minimum cost action, minimised over all the possible actions (u(k))
                    min_cost_index = np.argmin(self.q_table[int(x_values[k] + self.x_limit), int(x_values[k-1] + self.x_limit), int(u_values[k-1] + self.u_limit)])
                    u_values[k] = min_cost_index - (self.u_limit)  # Does action corresponding to minimum cost

                    # Calculates and stores cost at each time step for the particular 'a' and 'b' combination
                    self.cost[combination_index][k] = x_values[k]**2 + u_values[k]**2

                    # Basically limits x to x_limit and -x_limit for next state, and updates next state
                    x_values[k+1] = min(max(a * x_values[k] + b * u_values[k], -self.x_limit), self.x_limit)

                # Plots either trajectory or error over time steps
                if option == "trajectory":
                    plt.plot(range(self.time_steps + 1), x_values)  # plot the given trajectory for a single combination of 'a' and 'b' value
                elif option == "error":
                    plt.plot(range(self.time_steps), self.cost[combination_index])  # plot the cost for the trajectory
                
                # Increment combination index by 1
                combination_index += 1

        plt.ion()  # turn on interactive mode
        plt.pause(0.01)  # allow time for GUI to load
        plt.show()
        

    def run_multiple_batches(self, batch_number_until_plot=100, option='trajectory'):
        """Runs the specified number of batches, each batch consisting of multiple episodes."""

        for i in range(self.number_of_batches):
            if i > 10:
                self.epsilon = 0.5  # switches to epsilon greedy policy, want it to explore alot
            self.run_one_batch()

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

    agent = SimpleDiscreteQLearning(number_of_batches=5000)

    # Fix random seed
    random.seed(1000)

    # Plot trajectory graph
    agent.run_multiple_batches(batch_number_until_plot=10, option = 'trajectory')
    # Plot error graph
    agent.run_multiple_batches(batch_number_until_plot=10, option = 'error')