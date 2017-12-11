import random

import copy
from typing import Tuple

import numpy as np

from TrackParser import TrackParser

np.set_printoptions(linewidth=500)

class GameState:
    """
    A state object that represents where the agent is in the game
    """
    def __init__(self, current_position: tuple, current_velocity: tuple):
        self.current_position = current_position
        self.current_velocity = current_velocity
        # self.reward = reward

    def value(self):
        """
        Helper function to identify the state
        """
        return self.current_position[0], self.current_position[1], self.current_velocity[0], self.current_velocity[1]

    def __str__(self):
        return str((self.current_position[0], self.current_position[1], self.current_velocity[0], self.current_velocity[1]))

    def __hash__(self):
        return self.current_position.__hash__() + self.current_velocity.__hash__()

    def __eq__(self, other):
        try:
            return self.current_position == other.current_position and self.current_velocity == self.current_velocity
        except Exception:
            return False


class Game:

    def __init__(self, path_to_track_file: str, success_chance=.8, crash_restart=False) -> None:
        """
        Initializes the game with the specified parameters, including which track to use.
        Uses the Track parser to parse the track.

        Sets up the start of the game and needed information for the game object like possible start locations.
        """
        self.track = TrackParser.parse_track(path_to_track_file)
        self.possible_starts = self.get_possible_start_positions(self.track)
        self.track_dimension_y = len(self.track)
        self.track_dimension_x = len(self.track[0])

        self.success_chance = success_chance

        # How bad it is to crash, nearest if False, restart if True
        self.crash_restart = crash_restart

        # position (y, x)
        self.current_position = (0, 0)
        self.start_position = (0, 0)

        # Movement Variables (y, x)
        self.velocity = (0, 0)
        # self.acceleration = (0, 0)

        # init game to be ready.
        self.start()

    def get_possible_start_positions(self, track: np.ndarray) -> list:
        """
        Helper function to get all possible start locations.
        """
        tuple_indices = np.where(track == 1)
        results = []
        for y, x in zip(tuple_indices[0], tuple_indices[1]):
            results.append((y, x))

        return results

    def set_start_position(self) -> None:
        """
        Sets the start position for the game to a random start location
        :return:
        """
        start_index = random.randint(0, len(self.possible_starts)-1)
        self.start_position = self.possible_starts[start_index]

    def set_current_position(self, position: tuple) -> None:
        """
        Helper function to the current position on the game board.
        """
        self.current_position = copy.copy(position)

    def set_state(self, current_position, current_velocity):
        """
        Sets the state of the game, position and velocity.

        Used by Value Iteration
        """
        self.current_position = current_position
        self.velocity = current_velocity

    def start(self):
        """
        Set up the game to be the start of the game!
        """
        self.set_start_position()
        self.set_current_position(self.start_position)
        self.velocity = (0, 0)
        self.acceleration = (0, 0)

    def get_valid_states(self):
        """
        Returns a list of valid states, include all possible velocities that go with them.

        This is a very large number of states for these boards.

        Used by Value Iteration
        """
        tuple_indices = np.where(self.track != -1)
        position_results = []
        for y, x in zip(tuple_indices[0], tuple_indices[1]):
            position_results.append((y, x))

        actual_results = []
        for position in position_results:
            for y_velocity in range(-5, 6):
                for x_velocity in range(-5, 6):
                    state = GameState(position, (y_velocity,x_velocity))
                    actual_results.append(state)

        return actual_results

    def take_action_with_success_rate(self, action: tuple, success_rate: float) -> Tuple:
        """
        Action loop for the game with a given rate of action success.
        Given an action, validate the action and then perform the action with some degree of guaranteed
        success rate.

        Used by VI since it knows the model and accounts for it else where.
        """
        # Action is (a_y, a_x) -> Acceleration in y and x directions
        acceleration = self.valididate_accelearation(action)

        # acceleration is applied to velocity BEFORE position update.
        # 80% chance of actually applying acceleration (successful action)
        velocity_and_if_successful_update = self.update_velocity(self.velocity, acceleration, success_rate)
        self.velocity = velocity_and_if_successful_update[0]
        action_successful = velocity_and_if_successful_update[1]

        # LOGIC for MOVING
        # UPDATES the current position based on the rules of movement and the track.
        # returns (position, velocity and reward)
        position_velocity_reward = self.update_position(self.current_position, self.velocity, self.crash_restart)

        self.current_position = position_velocity_reward[0]
        self.velocity = position_velocity_reward[1]

        return GameState(self.current_position, self.velocity), position_velocity_reward[2], action_successful


    def take_action(self, action: tuple) -> Tuple:
        """
        Main Action loop for the game!
        Given an action, validate the action and then perform the action in accordance to all the rules!

        returns GameState and reward tuple
        """
        # Action is (a_y, a_x) -> Acceleration in y and x directions
        acceleration = self.valididate_accelearation(action)

        # acceleration is applied to velocity BEFORE position update.
        # 80% chance of actually applying acceleration (successful action)
        velocity_and_if_successful_update = self.update_velocity(self.velocity, acceleration, self.success_chance)
        self.velocity = velocity_and_if_successful_update[0]
        action_successful = velocity_and_if_successful_update[1]

        # LOGIC for MOVING
        # UPDATES the current position based on the rules of movement and the track.
        # returns (position, velocity and reward)
        position_velocity_reward = self.update_position(self.current_position, self.velocity, self.crash_restart)

        self.current_position = position_velocity_reward[0]
        self.velocity = position_velocity_reward[1]

        return GameState(self.current_position, self.velocity), position_velocity_reward[2], action_successful

    def update_position(self, current_position: tuple, velocity:tuple, restart: bool) -> tuple:
        """
        Updates the current position of the Agent based on velocity.
        Ensures that the resulting state is a valid state, that is on the track!
        """
        new_position = []
        for current_position_i, velocity_i in zip(current_position, velocity):
            new_position.append(current_position_i + velocity_i)

        crashed = False
        # Check validity of position (Bounds movement to the board)
        new_position = self.bound_position(self.track_dimension_y, self.track_dimension_x, new_position)

        new_y = new_position[0]
        new_x = new_position[1]

        # check for finished condition
        finished = False
        track_position_value = self.track[new_y][new_x]
        if track_position_value == 2:
            finished = True
            pass
        elif track_position_value == 1:
            # nothing happens
            pass
        elif self.track[new_y][new_x] == -1:
            crashed = True

        # apply reward rules
        if crashed and not restart:
            position = self.find_nearest_valid_track_placement(self.track, tuple(new_position))
            new_velocity = (0, 0)
            reward = -1
        elif crashed and restart:
            position = self.start_position
            new_velocity = (0, 0)
            reward = -1
        else:
            position = new_position
            new_velocity = velocity
            if finished:
                reward = 0
            else:
                reward = -1

        # return GameState(position, new_velocity, reward)
        return tuple(position), new_velocity, reward

    def update_velocity(self, velocities, accelerations: tuple, success_chance: float) -> tuple:
        """
        Velocity is bounded as y,x element of {+-5, +-5}

        Chance of actually applying acceleration (successful action)
        """

        # chance of actually applying acceleration (successful action)
        apply_move_chance = random.random()
        if apply_move_chance > success_chance:
            # print("NOT CHANGING VELOCITY")
            return velocities, False


        result_velocity = []

        for velocity_i, acceleration_i in zip(velocities, accelerations):
            result_velocity.append(velocity_i + acceleration_i)

        for index in range(len(result_velocity)):
            if result_velocity[index] > 5:
                result_velocity[index] = 5

            elif result_velocity[index] < -5:
                result_velocity[index] = -5

        return tuple(result_velocity), True

    def valididate_accelearation(self, accelerations: tuple) -> tuple:
        """
        Validates acceleration inputs
        Valid values are a{y,x} element of {-1,0,1}
        """
        accelerations = list(accelerations)
        for index in range(len(accelerations)):
            acceleration = accelerations[index]
            if acceleration != -1 and acceleration != 0 and acceleration != 1:
                accelerations[index] = 0

        return tuple(accelerations)

    def find_nearest_valid_track_placement(self, track: np.ndarray, position: tuple):
        """
        Finds the nearest point on the map that is a valid place on the track.
        Uses a radial grid search, in a box formation, looking for valid track positions.
        """
        track_value = track[position[0]][position[1]]
        radius = 1
        nearest_valid_position = None

        track_dimension_y = len(track)
        track_dimension_x = len(track[0])

        while nearest_valid_position is None:
            # generate new possible positions
            up = [position[0] - radius, position[1]]
            down = [position[0] + radius, position[1]]
            left = [position[0], position[1] - radius]
            right = [position[0], position[1] + radius]
            upper_left = [position[0] - radius, position[1] - radius]
            upper_right = [position[0] - radius, position[1] + radius]
            bottom_left = [position[0] + radius, position[1] - radius]
            bottom_right = [position[0] + radius, position[1] + radius]

            possible_positions = [up, down, left, right, upper_left, upper_right, bottom_left, bottom_right]

            # Bound all positions
            for index in range(len(possible_positions)):
                # Check validity of position (Bounds movement to the board)
                possible_position = possible_positions[index]
                possible_position = self.bound_position(track_dimension_y, track_dimension_x, possible_position)
                possible_positions[index] = possible_position

            for possible_position in possible_positions:
                track_pos_value = track[possible_position[0]][possible_position[1]]
                if track_pos_value == 0 or track_pos_value == 1:
                    nearest_valid_position = copy.copy(possible_position)
                    break

            radius += 1
        return nearest_valid_position


    def bound_position(self, track_dimension_y: int, track_dimension_x: int, position: list):
        """
        Bounds all movement to be some position on the grid. This prevents values from
        trying to leave the grid world space.
        """
        if position[0] >= track_dimension_y:
            position[0] = track_dimension_y - 1

        elif position[0] < 0:
            position[0] = 0

        if position[1] >= track_dimension_x:
            position[1] = track_dimension_x - 1

        elif position[1] < 0:
            position[1] = 0

        return position

    def print_game_board(self):
        """
        Prints the game board.
        """
        track = np.copy(self.track)
        track[self.current_position[0]][self.current_position[1]] = 5
        print(track)

    def get_current_state(self) -> GameState:
        """
        returns the current state in a game state object
        """
        return GameState(self.current_position, self.velocity)

    def is_goal(self, game_state: GameState) -> bool:
        """
        returns if the state given is a goal state.
        """
        track_value = self.track[game_state.current_position[0]][game_state.current_position[1]]
        if track_value == 2:
            return True
        else:
            return False





# x = np.array([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
#        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  2.,  2.,  2.,  2., -1.],
#        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1.],
#        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1.],
#        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1.],
#        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1.],
#        [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
#        [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
#        [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
#        [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
#        [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]])

# game = Game('tracks/L-track.txt', success_chance=1)
# # print(game.get_possible_start_positions(x))
# # print(game.find_nearest_valid_track_placement(x, (0, 1)))
# game.print_game_board()
# game.take_action((0, 1))
# game.print_game_board()
# game.take_action((0, 1))
# game.print_game_board()
# game.take_action((0, 1))
# game.print_game_board()
# game.take_action((0, 1))
# game.print_game_board()

# game = Game('tracks/L-track.txt', success_chance=1)
# print(len(game.get_valid_states()))
# print(game.get_possible_start_positions(x))
# print(game.find_nearest_valid_track_placement(x, (0, 1)))
# game.print_game_board()
# game_state = game.take_action((-1, 1))
# game.print_game_board()
# game.take_action((-1, 1))
# game.print_game_board()
# game.take_action((-1, 1))
# game.print_game_board()
# game.take_action((-1, 1))
# game.print_game_board()

# print(game.is_goal(GameState((2, 35), (0, 0))))
