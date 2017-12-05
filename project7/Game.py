""" Created by Max 12/2/2017 """
import random

import copy
import numpy as np

from TrackParser import TrackParser


class Game:

    def __init__(self, path_to_track_file: str) -> None:
        self.track = TrackParser.parse_track(path_to_track_file)
        self.possible_starts = self.get_possible_start_positions(self.track)

        # position (y, x)
        self.current_position = (0, 0)
        self.start_position = (0, 0)

        # Movement Variables (y, x)
        self.velocity = (0, 0)
        self.acceleration = (0, 0)

        # init game to be ready.
        self.start()

    def get_possible_start_positions(self, track: np.ndarray) -> list:
        tuple_indices = np.where(track == 1)
        results = []
        for y, x in zip(tuple_indices[0], tuple_indices[1]):
            results.append((y, x))

        return results

    def set_start_position(self) -> None:
        start_index = random.randint(0, len(self.possible_starts)-1)
        self.start_position = self.possible_starts[start_index]

    def set_current_position(self, position: tuple) -> None:
        self.current_position = copy.copy(position)

    def start(self):
        self.set_start_position()
        self.set_current_position(self.start_position)
        self.velocity = (0, 0)
        self.acceleration = (0, 0)


    def take_action(self, action: tuple) -> GameState:
        # Action is (a_y, a_x) -> Acceleration in y and x directions
        # Valid values are a{y,x} element of {-1,0,1}

        # Velocity is bounded as y,x element of {+-5, +-5}

        # acceleration is applied to velocity BEFORE position update.
        # 80% chance of actually applying acceleration (successful action)

        # LOGIC for MOVING
        # UPDATES the current position based on the rules of movement and the track.

        pass

    def ger_reward(self) -> int:
        pass


class GameState:
    def __init__(self, current_position, reward):
        self.current_position = current_position
        self.reward = reward



x = np.array([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  2.,  2.,  2.,  2., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -1.],
       [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]])

game = Game('tracks/L-track.txt')
print(game.get_possible_start_positions(x))


