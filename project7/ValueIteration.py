import copy
import pprint

import sys

from Game import Game
from Game import GameState


class ValueIteration:
    def __init__(self, game):
        """
        Init Variables for tracking the V_S values and the Q values.
        Init possible actions as well.
        """
        self.game = game

        self.gamma = .999

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
        """
        Main work horse for the VI algorithm.

        after initialization of everything so that all states are known and stored in V and Q

        start updating V and Q until the largest difference in V and V_last is less than epsilon.
        """
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

        return policy, update_cycle

    def update_q_iteration(self, game, state: tuple, action, q, gamma, v_s_last):
        """
        Updates the Q values for VI based on
        R(s,a) + gamma * sum( T(s,a,s') V_t-1(s)) for all s' in s
        """
        game.set_state((state[0], state[1]), (state[2], state[3]))

        state_reward_success = game.take_action_with_success_rate(action, 1)
        new_state = state_reward_success[0]
        reward = state_reward_success[1]
        if new_state is not None:
            q[state][action] = reward + gamma * self.sum_transition(game, state, action, v_s_last)
        return q

    def sum_transition(self, game: Game, state: tuple, action, v_s_last):
        """
        create the sum for the possible outcomes states given an action pair.
        the only two outcomes are that the action works or doesn't.
        as a result 2 s' need to be considered.
        The action is successful 80 percent of the time and not 20 thus the probability values infront of the updates.
        """

        sum_value = 0


        # S prime is after a successful transition with action
        game.set_state((state[0], state[1]), (state[2], state[3]))
        state_reward_success = game.take_action_with_success_rate(action, 1)

        new_state = state_reward_success[0]
        reward = state_reward_success[1]
        successful = state_reward_success[2]
        sum_value += .8 * v_s_last[new_state.value()]

        # S prime is after a failed transition
        game.set_state((state[0], state[1]), (state[2], state[3]))
        state_reward_success = game.take_action_with_success_rate(action, 0)

        new_state = state_reward_success[0]
        reward = state_reward_success[1]
        successful = state_reward_success[2]
        sum_value += .2 * v_s_last[new_state.value()]


        return sum_value

    def init_policy(self, v_s):
        """
        Inits the policy table
        """
        policy = {}
        for state in v_s:
            policy[state] = None
        return policy

    def init_q(self, v_s, actions):
        """
        Init the Q table
        """
        q = {}
        for state in v_s:
            q[state] = {}
            for action in actions:
                q[state][action] = -sys.maxsize
        return q

    def arg_max_iteration_version(self, current_state: tuple, q):
        """
        Get the argmax in the Q table
        """
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
        """
        The check to see if we are done.
        we stop when no difference in the table is larger than epsilon. 
        """
        if is_first_round:
            return False
        for state in v_s:
            if v_s[state] is None:
                continue
            if state not in v_s_last:
                return False
            if abs(v_s[state] - v_s_last[state]) > epsilon:
                print(abs(v_s[state] - v_s_last[state]))
                return False
        return True


    def execute_policy(self, policy, show=False):
        """
        Used for executing a learned memory with no updates

        """
        self.game.start()
        game = self.game

        current_state = game.get_current_state()

        num_steps_taken = 0

        while not game.is_goal(current_state):
            # action = self.select_action(policy, current_state, 0.01)
            action = policy[current_state.value()]

            new_state_and_reward = game.take_action(action)
            new_state = new_state_and_reward[0]
            reward = new_state_and_reward[1]

            current_state = copy.copy(new_state)

            if show:
                game.print_game_board()
                print()

            num_steps_taken += 1

        return num_steps_taken

# game = Game('tracks/L-track.txt', success_chance=.8)
# vi = ValueIteration(game)
# policy = vi.value_iteration()
# print(policy)

# game = Game('tracks/R-track.txt', success_chance=.8)
# vi = ValueIteration(game)
# policy = vi.value_iteration()
# pprint.pprint(policy)

# game = Game('tracks/R-track.txt', success_chance=.8, crash_restart=True)
# vi = ValueIteration(game)
# policy = vi.value_iteration()
# pprint.pprint(policy)

