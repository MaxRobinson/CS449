from Game import Game
from ValueIteration import ValueIteration


def experiment(test_game, num_experiments):
    """
    Main experiment method that runs the Value Iteration experiments and prints results
    works by learning a model x number of times.

    the average number of moves per policy is then created and averaged per experiment

    prints and returns the average number of episodes to reach the goal along with the learned policy.
    """

    average_number_of_moves_with_policy = []
    for x in range(num_experiments):
        # Learn Policy
        vi = ValueIteration(test_game)
        policy_and_num_iterations = vi.value_iteration()
        policy = policy_and_num_iterations[0]
        print(policy)

        avg_num_steps = 0
        for itter in range(100):
            num_steps = vi.execute_policy(policy)
            avg_num_steps += num_steps

        avg_num_steps /= 100.0

        average_number_of_moves_with_policy.append(avg_num_steps)

    total_average_num_steps = sum(average_number_of_moves_with_policy) / num_experiments
    print("Total Average Number of Steps: {}".format(total_average_num_steps))

    return total_average_num_steps

# <editor-fold desc="Experiments">


game = Game('tracks/L-track.txt', success_chance=.8)
experiment(game, 10)

game = Game('tracks/R-track.txt', success_chance=.8)
experiment(game, 10)

game = Game('tracks/R-track.txt', success_chance=.8, crash_restart=True)
experiment(game, 10)




# </editor-fold>


