from __future__ import division
import csv, math, random, sys
import datetime

from customCsvReader import CustomCSVReader

__author__ = 'Max'


class Knn:
    def __init__(self, k, classification=True):
        """
        Constructor
        :param k: int, number of nearest neighboors to look at
        :param classification: boolean, true if classification, else regression
        """
        self.k = k
        self.classification = classification

    # <editor-fold desc="Learn">
    def learn(self, dataset, query):
        """
        Given a model of data points and a query point, return the predicted result.
        Will do classification or regression based on set flag.

        inintalize k smallest distances it max dist
        for each data point in the model, calculate the distance, and add that to the distance list.

        process the list of smallest distances according to classification or regression.

        :param dataset: list of data points, model
        :param query: data point, query point.
        :return: for classification a label, for regression a value prediction.
        """
        # distances = []
        k_smallest_distances = [(sys.maxsize, []) for x in range(self.k)]

        for datapoint in dataset:
            datapoint_features_only = datapoint[:-1]
            distance_of_features = self.distance(datapoint_features_only, query)
            k_smallest_distances = self.add_to_smallest_distances(k_smallest_distances, distance_of_features, datapoint)

        if self.classification:
            return self.process_for_classification(k_smallest_distances)
        else:
            return self.process_for_regression(k_smallest_distances)

    def test(self, dataset, query_set):
        """
        A helper method to process many query points with the same data model.
        for each query point, get the predicted value, and add it to a list.
        return list of predictions

        :param dataset: list of data points, model
        :param query_set: list of datapoints, query points
        :return: list of predictions, either class labels, or floats.
        """
        results = []
        for item in query_set:
            results.append(self.learn(dataset, item))
        return results

    # </editor-fold>

    # <editor-fold desc="Helpers">
    def add_to_smallest_distances(self, k_smallest_distances, distance_of_features, datapoint):
        """
        A helper function to add a distance to the list of smallest distances only if it smaller than
        the other distances currnetly in the list.

        compare the distance to the largest distance in the list
        if smaller, replace the largest value in the list.
        sort the list.

        Ensure the largest item is always the last item.

        :param k_smallest_distances: list of tuples(distance of feature, datapoint).
        :param distance_of_features: float distance
        :param datapoint: datapoint with corresponding to the distance
        :return: list of smallest distances (tuples)
        """
        if distance_of_features <= k_smallest_distances[-1][0]:
            k_smallest_distances[-1] = (distance_of_features, datapoint)
            k_smallest_distances.sort(key=lambda tup: tup[0])

        return k_smallest_distances

    def distance(self, datapoint, query):
        """
        Calculates the Euclidean distance between a datapoint in the training set and a query point.

        :param datapoint: A feature Vector that is a datapoint in the training data
        :param query: a feature vector that we are looking to predict the value for.
        :return: the Euclidean distance between datapoints.
        """
        dist_sum = 0
        for datapoint_feature, query_feature in zip(datapoint, query):
            dist_sum += self.feature_distance(datapoint_feature, query_feature)

        dist = math.sqrt(dist_sum)

        return dist

    def feature_distance(self, datapoint_feature, query_feature):
        """
        Calculates the distance between two features. Euclidean distance
        :param datapoint_feature: a feature, float
        :param query_feature: a query feature, flaot
        :return: distance
        """
        feature_diff = datapoint_feature - query_feature
        feature_diff *= feature_diff
        return feature_diff

    # </editor-fold>

    def process_for_regression(self, k_nearest):
        """
        For the regression problem, process the k_nearest and provides a predicted value.
        Does a simple average over all of the k_nearest neighbors.

        :param k_nearest: a list of tuples where tuples are (distance, feature_vector)
        :return: the average value of the k nearest neighbors.
        """
        y_sum = 0
        for dist, features in k_nearest:
            y_sum += features[-1]

        average = y_sum / len(k_nearest)

        return average

    def process_for_classification(self, k_nearest):
        """
        For the regression problem, process the k_nearest and provides a predicted value.

        Calculates a simple majority vote

        :param k_nearest: a list of tuples where tuples are (distance, feature_vector)
        :return: the class label with the majority votes for that class.
        """
        labels = {}
        for dist, feature_vector in k_nearest:
            classLabel = feature_vector[-1]
            if classLabel not in labels:
                labels[classLabel] = 1
            else:
                labels[classLabel] += 1
        return max(labels, key=labels.get)


# <editor-fold desc="Test">
# data = CustomCSVReader.read_file("data/machine.data.new.txt")
#
#
# best_k = get_best_k_for_data_mse(data)
# print(best_k)
# </editor-fold>
