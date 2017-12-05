""" Created by Max 12/2/2017 """

from TrackParser import TrackParser

class Track:
    def __init__(self, path_to_track_file):
        TrackParser.parse_track(path_to_track_file)
