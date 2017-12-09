""" Created by Max 12/9/2017 """
import pickle

import copy
import random

import sys
from pprint import pprint

from Game import Game, GameState

class QLearning:
    def __init__(self, game, epsilon=.99, alpha=.99, decay_rate=.99, gamma=.9, num_episodes=300, save_rate=20):
        self.game = game
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes
        self.save_rate = save_rate

        self.q = {}
        self.num_moves_per_episode = []

        self.num_random_moves_per_episode = []
        self.num_random_moves = 0

        self.possible_actions = ((-1, -1),
                                 (-1, 0),
                                 (-1, 1),
                                 (0, -1),
                                 (0, 0),
                                 (0, 1),
                                 (1, -1),
                                 (1, 0),
                                 (1, 1))

    def learn(self) -> dict:
        episode_count = 1
        while episode_count <= self.num_episodes:
            # Decay learning rate and exploration over time
            # self.alpha *= self.decay_rate
            if self.epsilon > 0.01:
                self.epsilon *= self.decay_rate
            else:
                self.epsilon = 0.01

            # self.epsilon *= self.decay_rate

            self.num_random_moves = 0

            # print("Alpha: {}".format(self.alpha))
            # print("Epsilon: {}".format(self.epsilon))

            self.game.start()
            q_and_num_steps = self.q_learning(self.game, self.q, self.epsilon, self.alpha, self.gamma)

            self.q = q_and_num_steps[0]
            self.num_moves_per_episode.append(q_and_num_steps[1])
            self.num_random_moves_per_episode.append(self.num_random_moves)

            if episode_count % self.save_rate == 0:
                # self.save(episode_count)
                print("Episode Count: {}".format(episode_count))

            episode_count += 1

        return self.q

    def q_learning(self, game: Game, q, epsilon, alpha, gamma) -> tuple:
        # init state
        current_state = game.get_current_state()

        num_steps_taken = 0

        while not game.is_goal(current_state):
            action = self.select_action(q, current_state, epsilon)
            new_state_and_reward = game.take_action(action)
            new_state = new_state_and_reward[0]
            reward = new_state_and_reward[1]
            # update Q
            q = self.update_q(q, new_state, current_state, action, reward)
            current_state = copy.copy(new_state)

            num_steps_taken += 1



        return q, num_steps_taken

    def update_q(self, q, new_state: GameState, previous_state: GameState, action_taken: tuple, reward: int) -> dict:
        """
        Update q based on:
        q[s,a] = (1-alpha)Q[s,a] + alpha(r + gamma* max_args(Q[s'])
        """
        previous_state_hash = previous_state.value()

        if action_taken is None:
            action_taken = ("end", "end")

        if previous_state_hash not in q:
            q[previous_state_hash] = {}

        if action_taken not in q[previous_state_hash]:
            q[previous_state_hash][action_taken] = reward

        current_q_value = q[previous_state_hash][action_taken]
        q[previous_state_hash][action_taken] = \
            current_q_value + self.alpha * (reward + (self.gamma * self.get_max_value(new_state.value(), q)) - current_q_value)

        return q


    def get_max_value(self, state, q):
        max_value = -sys.maxsize

        if state not in q:
            return 0
        for action in q[state]:
            if q[state][action] > max_value:
                max_value = q[state][action]

        return max_value

    def select_action(self, q, current_state, epsilon) -> tuple:
        role = random.random()
        action = None

        if role < epsilon:
            action = self.select_random_move(self.possible_actions)
        else:
            action = self.arg_max(current_state.value(), q)

        if action is None:
            action = self.select_random_move(self.possible_actions)

        return action

    def select_random_move(self, possible_actions: tuple) -> tuple:
        self.num_random_moves += 1
        role = random.randint(0, len(possible_actions)-1)
        return possible_actions[role]

    def arg_max(self, current_state, q):
        max_arg = None
        max_value = -sys.maxsize

        if current_state not in q:
            return None

        for action in q[current_state]:
            if q[current_state][action] > max_value:
                max_arg = action
                max_value = q[current_state][action]

        return max_arg

    def save(self, count):
        output = open('q_save_file_normal_large_world{}.txt'.format(count), 'wb')
        pickle.dump(self.q, output)
        output.close()


    def execute_policy(self, q, show=False):
        """
        Used for executing a learned memory with no updates

        """
        self.game.start()
        game = self.game

        current_state = game.get_current_state()

        num_steps_taken = 0

        while not game.is_goal(current_state):
            action = self.select_action(q, current_state, 0.01)
            new_state_and_reward = game.take_action(action)
            new_state = new_state_and_reward[0]
            reward = new_state_and_reward[1]
            # update Q
            # q = self.update_q(q, new_state, current_state, action, reward)
            current_state = copy.copy(new_state)

            if show:
                game.print_game_board()
                print()

            num_steps_taken += 1

        return q, num_steps_taken





# Tests
test_game = Game('tracks/L-track.txt', success_chance=1)

q_learning = QLearning(test_game, num_episodes=5000, alpha=.7, gamma=.7)
q = q_learning.learn()

# pprint(q)
print(q_learning.num_moves_per_episode)
print(q_learning.num_random_moves_per_episode)

result = q_learning.execute_policy(q, True)

print(result[1])
# print(len(q.keys()))



# test_game = Game('tracks/R-track.txt', success_chance=1)
#
# q_learning = QLearning(test_game, num_episodes=5000, alpha=.7, gamma=.7)
# q = q_learning.learn()
#
# # pprint(q)
# print(q_learning.num_moves_per_episode)
# print(q_learning.num_random_moves_per_episode)
#
# result = q_learning.execute_policy(q, True)
#
# print(result[1])
# # print(len(q.keys()))
