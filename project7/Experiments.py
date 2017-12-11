import pprint

import sys

from Game import Game
from Game import GameState
from QLearning import QLearning

import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def experiment(test_game, num_experiments, file_name, num_episodes=500, alpha=.99, gamma=.9, epsilon=.9, decay_rate=.99):
    """
    Main experiment method that runs the Q-Learning experiments and returns prints and draws the needed diagrams.
    works by learning a model x number of times and then compiling the number of steps per epoch for experiment
    These are then averaged and used to create a graph.

    A policy is then also chosen to give an average number of steps needed to reach the goal metric.
    """

    list_of_moves_per_experiment = []
    policies = []
    for x in range(num_experiments):
        # Learn model
        q_learning = QLearning(test_game, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, decay_rate=decay_rate)
        q = q_learning.learn()
        policies.append(q)

        num_moves = q_learning.num_moves_per_episode
        list_of_moves_per_experiment.append(num_moves)


    np.array(list_of_moves_per_experiment)
    moves_per_epoc_number = np.sum(list_of_moves_per_experiment, axis=0)
    moves_per_epoc_number = moves_per_epoc_number / num_experiments


    # get Average number of steps when executing.
    q_learning = QLearning(test_game, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, decay_rate=decay_rate)
    avg_num_steps = 0
    for itter in range(100):
        num_steps = q_learning.execute_policy(policies[num_experiments-1])
        avg_num_steps += num_steps[1]

    avg_num_steps /= 100.0

    generate_validation_curves(np.arange(num_episodes), moves_per_epoc_number, None, "Number of steps", None,
                               x_axis_label="Epoc Number", y_axis_label="Average Path Length",
                               file_name=file_name)

    return avg_num_steps, policies[num_experiments-1]


def generate_validation_curves(x_axis_values, values_line_1, values_line_2, label_1, label_2, x_axis_label="",
                               y_axis_label="Average Epoc Path Length", title="", file_name=""):
    """
    A helper function to draw graphs and save them to a file.
    """
    plt.plot(x_axis_values, values_line_1, '-', label=label_1)
    # plt.plot(x_axis_values, values_line_2, '-', label=label_2)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    plt.legend(loc='best')

    if file_name != "":
        plt.savefig(file_name)

    plt.show()
    plt.close()


# <editor-fold desc="Experiments">


game = Game('tracks/L-track.txt', success_chance=.8)
result = experiment(game, 10, "diagrams/L-Track-Easy-1", num_episodes=5000, alpha=.7, gamma=.7, epsilon=.99, decay_rate=.9)
pprint.pprint(result)

game = Game('tracks/R-track.txt', success_chance=.8)
result = experiment(game, 10, "diagrams/R-Track-Easy-1", num_episodes=2000, alpha=.7, gamma=.7, epsilon=.99, decay_rate=.9)
pprint.pprint(result)

game = Game('tracks/R-track.txt', success_chance=.8, crash_restart=True)
result = experiment(game, 10, "diagrams/R-Track-Harsh-update", num_episodes=5000, alpha=.7, gamma=.7, epsilon=.1)
pprint.pprint(result)

# </editor-fold>


