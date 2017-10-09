""" Created by Max 10/8/2017 """
import math
import sys
from customCsvReader import CustomCSVReader


"""
This file was used to calculate the Max MSE that exists in the data sets for comparison of how well
the k-NN regression algorithm did when comparing MSE
"""

def distance(datapoint, query):
    """
    Calculates the Euclidean distance between a datapoint in the training set and a query point.

    :param datapoint: A feature Vector that is a datapoint in the training data
    :param query:
    :return:
    """
    dist_sum = 0
    for datapoint_feature, query_feature in zip(datapoint, query):
        dist_sum += feature_distance(datapoint_feature, query_feature)

    dist = math.sqrt(dist_sum)

    return dist

all_data = CustomCSVReader.read_file("D:\\Documents\\JHU-Masters\\605.449 Intro to Machine Learning\\projects\\project3\\data\\machine.data.new.txt", float)
# all_data = CustomCSVReader.read_file("D:\\Documents\\JHU-Masters\\605.449 Intro to Machine Learning\\projects\\project3\\data\\forestfires.data.new.txt", float)

max_distance = 0
max_dist_points = ()
for datapoint_i in all_data:
    for datapoint_j in all_data:
        dist = (datapoint_i[-1] - datapoint_j[-1])**2
        if dist > max_distance:
            max_distance = dist
            max_dist_points = (datapoint_i, datapoint_j)

print(max_distance)
print(max_distance**2)

print((max_dist_points[0][-1] - max_dist_points[1][-1])**2)

print(max_dist_points[0])
print(max_dist_points[1])


