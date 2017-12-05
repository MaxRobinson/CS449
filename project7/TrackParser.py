""" Created by Max 12/2/2017 """
import pprint

import numpy as np


class TrackParser:

    @staticmethod
    def parse_track(path_to_track_file: str) -> np.ndarray:
        track = None
        with open(path_to_track_file, 'r') as track_file:
            lines = track_file.readlines()
            dimensions_str = lines[0]
            dims = dimensions_str.split(',')

            # (Y,X) coords.
            track = np.zeros((int(dims[0]), int(dims[1])))

            for line_index in range(1, len(lines)):
                line = lines[line_index]
                for char_index in range(len(line) - 1):
                    track_value = TrackParser.get_char_value(line[char_index])
                    track[line_index-1][char_index] = track_value

        return track

    @staticmethod
    def get_char_value(char: str):
        if char == '#':
            return -1
        elif char == '.':
            return 0
        elif char == 'S':
            return 1
        elif char == 'F':
            return 2
        else:
            return -1


# np.set_printoptions(linewidth=500)
# pprint.pprint(TrackParser.parse_track("tracks/L-track.txt"), width=500)

