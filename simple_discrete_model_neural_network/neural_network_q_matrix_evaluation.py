# This file uses the create_evaluation_metrics file under the 'evaluating_agent_performance' subfolder in order to output
# evaluation metrics for the neural_network_q_matrix case, as specified in the 'Evaluating RL algorithms' document.

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from neural_network_q_matrix import DQN  # import RL agent class
from evaluating_agent_performance.create_evaluation_metrics import CreateEvaluationMetrics  # import evaluation class


def print_statement(mean_vector, variance_vector):
    "Helper function for printing the mean and variance values."

    print(f'---------------------------------------------------------------')
    print(f'Total fraction of samples that converge to optimal policy: {mean_vector[0]}, variance of {variance_vector[0]}')
    print(f'Average number of time steps for convergence: {mean_vector[1]}, variance of {variance_vector[1]}')
    print(f'Average error per time step: {mean_vector[2]}, variance of {variance_vector[2]}')
    print(f'---------------------------------------------------------------')


def main_quantitative():
    "Quantitative evaluation metrics"

    # No. batches until convergence = X

    agent = DQN()  # create agent
    evaluation = CreateEvaluationMetrics(agent)  # create evaluation metric, passing in agent

    # Evaluate metrics on a single test epsiode, which contains samples of ALL combinations of unseen and seen failure modes.
    mean_vector, variance_vector = evaluation.create_evaluation_table_overall(number_batches=15000, episodes_per_batch=100, 
                                                                              seed_range=[x for x in range(2)], test_type='overall')
    # Output mean vector and std. vector
    print(f'Overall combination of failure modes')
    print_statement(mean_vector, variance_vector)

    # Evaluate metrics on a single test epsiode, which contains samples of only the trained-on/seen failure modes
    mean_vector, variance_vector = evaluation.create_evaluation_table_overall(number_batches=15000, episodes_per_batch=100, 
                                                                              seed_range=[x for x in range(2)], test_type='seen')
    # Output mean vector and covariance vector
    print(f'Trained-on combinations of failure modes')
    print_statement(mean_vector, variance_vector)

    # Evaluate metrics on a single test epsiode, which contains samples of only the unseen failure modes.
    mean_vector, variance_vector = evaluation.create_evaluation_table_overall(number_batches=15000, episodes_per_batch=100, 
                                                                              seed_range=[x for x in range(2)], test_type='unseen')
    # Output mean vector and covariance vector
    print(f'Unseen combinations of failure modes')
    print_statement(mean_vector, variance_vector)


def main_qualitative():
    "Qualitative evaluation metrics"

    # No. batches until convergence = X

    agent = DQN()  # create agent
    evaluation = CreateEvaluationMetrics(agent)  # create evaluation metric, passing in agent

    # Produce plots on a single test epsiode, which contains samples of ALL combinations of unseen and seen failure modes.
    evaluation.qualitative_evaluation_plots(number_batches=15000, episodes_per_batch=100, test_type='overall')

    # Produce plots on a single test epsiode, which contains samples of only the trained-on/seen failure modes
    evaluation.qualitative_evaluation_plots(number_batches=15000, episodes_per_batch=100, test_type='seen')

    # Produce plots on a single test epsiode, which contains samples of only the unseen failure modes.
    evaluation.qualitative_evaluation_plots(number_batches=15000, episodes_per_batch=100, test_type='unseen')


if __name__ == "__main__":
    # Call main_quantitative or main_qualitative, depending on which type of evaluation we want.

    main_quantitative()
    #main_qualitative()