# This file creates a class that takes in an agent object (e.g. Q Learning agent), and evaluates the defined algorithm in a fair
# and precisely defined manner, as stated in the 'Evaluating performance for RL algorithms' document.

import numpy as np
import random
import statistics

class CreateEvaluationMetrics:
    """
    This class takes in an agent object, and evaluates it's performance, as specified in the 'Evaluating performance for RL algorithms'
    document. The main 3 quantitative measures of performance will be calculated. The main method the agent class must implement is:

    simulate_single_test_epsiode(self): Runs a single test episode, under the current trained policy.
                                        Returns a cost and state matrix for each possible combination (one sample trajectory for a given 'a' 
                                        and 'b') at each time step. Size of each matrix is number_of_combinations * self.time_steps.

    The 3 quantitative measures of performance to be calculated are:
    1. Count the total fraction of samples that converge to the optimal policy, for a single test episode that is run AFTER the 
       last episode of the training.
    2. Count the average number of time steps required to reach convergence, for a single test episode that is run AFTER the last 
       episode of the training. (This only applies to samples that did converge)
    3. Measure the error/cost over each time step, and average over all time steps, for a single test episode that is run AFTER the 
       last episode of the training. This is averaged over all samples.

    Each algorithm is trained for a fixed number of episodes, and then the current trained algorithm is tested against the 
    above evaluation guidelines.

    This process is then repeated for a range of random seeds, which in itself has a fixed range. 
    This is to ensure that we get as unbiased an estimate as possible of the algorithm’s evaluation metrics, as there is some stochasticity 
    to the training itself (due to the randomness in exploration).

    The mean and variances of each individual quantitative evaluation metric can then be calculated, thus leading to the final 
    evaluation table of a specific RL algorithm - our final evaluation metric table.
    """


    def __init__(self, agent):
        self.agent = agent
        self.cost_matrix = []
        self.state_matrix = []


    def calculate_cost_and_state_for_single_test_episode(self):
        """Computes the cost and state matrix for a single test epsiode."""

        self.cost_matrix, self.state_matrix = self.agent.simulate_single_test_epsiode()


    def count_fraction_of_samples_that_converge_per_epsiode(self):
        """Counts the total fraction of samples that converge to the optimal policy, for a single test episode."""
        
        total_trajectories_that_converged = 0
        total_combinations_of_trajectories = len(self.agent.possible_a_vector) + len(self.agent.possible_b_vector)

        for row in self.state_matrix:
            if row[-1] == 0:
                total_trajectories_that_converged += 1
            else:
                pass

        total_fraction_that_converged = total_trajectories_that_converged / total_combinations_of_trajectories

        return total_fraction_that_converged


    def count_time_steps_for_convergence(self):
        """Counts the average number of time steps required for convergence."""

        time_steps_for_convergence = []

        for row in self.state_matrix:  # iterate over all trajectories
            current_time_step = 1  # reset counter
            if row[-1] == 0:  # only for states that converge
                 for state_at_single_time_step in row[1:]:  # ignore x[0]
                    if state_at_single_time_step == 0:
                        time_steps_for_convergence.append(current_time_step)
                    else:
                        pass
                    current_time_step += 1
            else:
                pass
        
        if time_steps_for_convergence is None:
            average_number_of_time_steps_for_convergence = None
        else:
            average_number_of_time_steps_for_convergence = sum(time_steps_for_convergence) / len(time_steps_for_convergence)

        return average_number_of_time_steps_for_convergence
        
    
    def average_cost_per_episode(self):
        """Measures the error/cost over each time step, averaged over all time steps and all samples."""

        return np.mean(self.cost_matrix)


    def create_evaluation_table(self, number_batches=5000, episodes_per_batch=100, seed_range=[x for x in range(100)]):
        """
        Each algorithm is trained for a fixed number of batches (number_batches) and episodes per batch (episodes_per_batch), and 
        then the current trained algorithm is tested against the above evaluation guidelines. 
        
        This process is then repeated for a range of random seeds, which in itself has a fixed range (seed_range).
        This is to ensure that we get as unbiased an estimate as possible of the algorithm’s evaluation metrics, as there is some 
        stochasticity to the training itself (due to the randomness in exploration).

        The mean and variances of each individual quantitative evaluation metric can then be calculated, thus leading to the final 
        evaluation table.
        """

        # Initialize the training parameters
        self.agent.number_of_batches = number_batches
        self.agent.number_of_episodes_per_batch = episodes_per_batch

        # Repeat for each seed in the seed range
        for seed_number in seed_range:
            # Set seed
            random.seed(seed_number)

            # Trains the agent as specified
            self.agent.run_multiple_batches_and_train()

            # Simulate a single test episode after training
            self.calculate_cost_and_state_for_single_test_episode()

            # Initialize lists for each metric
            metric_one = []
            metric_two = []
            metric_three = []

            # Compute each metric, and append to list
            metric_one.append(self.count_fraction_of_samples_that_converge_per_epsiode())
            metric_two.append(self.count_time_steps_for_convergence())
            metric_three.append(self.average_cost_per_episode())

        # Compute the mean and variance of each evaluation metric, and append to vector
        mean_vector = []
        variance_vector = []
        combined_metrics = [metric_one, metric_two, metric_three]

        for metric in combined_metrics:  # iterating over the three metrics
            mean_value = statistics.mean(value for value in metric if value is not None)
            try:
                variance_value = statistics.variance(value for value in metric if value is not None)

            # Error handling for the case where all values are None in one of the metric vectors
            except statistics.StatisticsError:
                variance_value = None

            mean_vector.append(mean_value)
            variance_vector.append(variance_value)

        return mean_vector, variance_vector