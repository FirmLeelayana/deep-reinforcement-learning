# This file uses the CreateEvaluationMetrics class under the 'evaluating_agent_performance' subfolder in order to output
# evaluation metrics, as specified in the 'Evaluating RL algorithms' document.

# Importing in the RL agent and evaluation classes
import DiscreteQLearning
from evaluating_agent_performance import CreateEvaluationMetrics


def main():
    agent = DiscreteQLearning()  # create agent
    evaluation = CreateEvaluationMetrics(agent)  # create evaluation metric, passing in agent
    mean_vector, variance_vector = evaluation.create_evaluation_table(number_batches=1000, episodes_per_batch=100, seed_range=[x for x in range(100)])

    # Output mean vector and covariance vector
    print(mean_vector, variance_vector)


if __name__ == "__main__":
    main()