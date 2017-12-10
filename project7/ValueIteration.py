""" Created by Max 12/9/2017 """
import copy

import sys

from Game import Game
from Game import GameState


class ValueIteration:
    def __init__(self, game):
        self.game = game

        self.gamma = .8

        self.q = {}
        self.v = {}
        self.possible_actions = ((-1, -1),
                                 (-1, 0),
                                 (-1, 1),
                                 (0, -1),
                                 (0, 0),
                                 (0, 1),
                                 (1, -1),
                                 (1, 0),
                                 (1, 1))

    def value_iteration(self):
        epsilon = .1

        states = self.game.get_valid_states()
        for state in states:
            if self.game.is_goal(state):
                self.v[state.value()] = 1
            else:
                self.v[state.value()] = 0

        v_last = copy.deepcopy(self.v)

        self.q = self.init_q(self.v, self.possible_actions)
        policy = self.init_policy(self.v)

        is_first_round = True
        update_cycle = 0
        while not self.v_s_comp_v_s_last_less_than_epsilon(v_last, self.v, epsilon, is_first_round):
            is_first_round = False
            v_last = copy.deepcopy(self.v)
            # count = 0
            for tuple_state in self.v:
                if self.game.is_goal(GameState((tuple_state[0], tuple_state[1]), (tuple_state[2], tuple_state[3]))):
                    policy[tuple_state] = None
                    # reward value
                    self.v[tuple_state] = 1
                else:
                    for action in self.possible_actions:
                        self.update_q_iteration(self.game, tuple_state, action, self.q, self.gamma, v_last)
                    policy[tuple_state] = self.arg_max_iteration_version(tuple_state, self.q)
                    self.v[tuple_state] = self.q[tuple_state][policy[tuple_state]]
                # print(count)
                # count += 1
            print(update_cycle)
            update_cycle += 1

        return policy

    def update_q_iteration(self, game, state: tuple, action, q, gamma, v_s_last):
        game.set_state((state[0], state[1]), (state[2], state[3]))

        state_reward_success = game.take_action(action)
        new_state = state_reward_success[0]
        reward = state_reward_success[1]
        if new_state is not None:
            q[state][action] = reward + gamma * self.sum_transition(game, state, self.possible_actions, v_s_last)
        return q

    def sum_transition(self, game: Game, state: tuple, actions, v_s_last):
        sum_value = 0
        for action in actions:
            game.set_state((state[0], state[1]), (state[2], state[3]))
            state_reward_success = game.take_action(action)
            new_state = state_reward_success[0]
            reward = state_reward_success[1]
            successful = state_reward_success[2]

            if new_state is None:
                print("WTF")

            if successful:
                sum_value += .8 * v_s_last[new_state.value()]
            else:
                sum_value += .2 * v_s_last[new_state.value()]

        return sum_value

    def init_policy(self, v_s):
        policy = {}
        for state in v_s:
            policy[state] = None
        return policy

    def init_q(self, v_s, actions):
        q = {}
        for state in v_s:
            q[state] = {}
            for action in actions:
                q[state][action] = -sys.maxsize
        return q

    def arg_max_iteration_version(self, current_state: tuple, q):
        max_arg = None
        max_value = -sys.maxsize
        if current_state not in q:
            return None
        for action in q[current_state]:
            if q[current_state][action] > max_value:
                max_arg = action
                max_value = q[current_state][action]

        return max_arg

    def v_s_comp_v_s_last_less_than_epsilon(self, v_s_last, v_s, epsilon, is_first_round):
        if is_first_round:
            return False
        for state in v_s:
            if v_s[state] is None:
                continue
            if state not in v_s_last:
                return False
            if abs(v_s[state] - v_s_last[state]) > epsilon:
                return False
        return True


game = Game('tracks/L-track.txt', success_chance=.8)
vi = ValueIteration(game)
policy = vi.value_iteration()
print(policy)

