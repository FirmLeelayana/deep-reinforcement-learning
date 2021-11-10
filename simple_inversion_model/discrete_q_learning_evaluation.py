# This file uses the create_evaluation_metrics file under the 'evaluating_agent_performance' subfolder in order to output
# evaluation metrics for the discrete_q_learning case, as specified in the 'Evaluating RL algorithms' document.

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from discrete_q_learning import DiscreteQLearning  # import RL agent class
from evaluating_agent_performance.create_evaluation_metrics import CreateEvaluationMetrics  # import evaluation class


def main():
    agent = DiscreteQLearning()  # create agent
    evaluation = CreateEvaluationMetrics(agent)  # create evaluation metric, passing in agent

    # Evaluate metrics on a single test epsiode, which contains samples of ALL combinations of unseen and seen failure modes.
    mean_vector, variance_vector = evaluation.create_evaluation_table_overall(number_batches=2500, episodes_per_batch=100, 
                                                                              seed_range=[x for x in range(10)], test_type='overall')
    # Output mean vector and covariance vector
    print(f'Overall -> Mean Vector: {mean_vector}, Variance Vector: {variance_vector}')

    # Evaluate metrics on a single test epsiode, which contains samples of only the trained-on/seen failure modes
    mean_vector, variance_vector = evaluation.create_evaluation_table_overall(number_batches=2500, episodes_per_batch=100, 
                                                                              seed_range=[x for x in range(10)], test_type='seen')
    # Output mean vector and covariance vector
    print(f'Seen -> Mean Vector: {mean_vector}, Variance Vector: {variance_vector}')

    # Evaluate metrics on a single test epsiode, which contains samples of only the unseen failure modes.
    mean_vector, variance_vector = evaluation.create_evaluation_table_overall(number_batches=2500, episodes_per_batch=100, 
                                                                              seed_range=[x for x in range(10)], test_type='unseen')
    # Output mean vector and covariance vector
    print(f'Unseen -> Mean Vector: {mean_vector}, Variance Vector: {variance_vector}')


if __name__ == "__main__":
    main()