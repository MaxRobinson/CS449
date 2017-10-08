from __future__ import division
import matplotlib.pyplot as plt
import csv, math, random, sys
import datetime
import numpy as np
import matplotlib.mlab as mlab

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

        :param dataset:
        :param query:
        :return:
        """
        # distances = []
        k_smallest_distances = [(sys.maxsize, []) for x in range(self.k)]

        for datapoint in dataset:
            datapoint_features_only = datapoint[:-1]
            distance_of_features = self.distance(datapoint_features_only, query)
            # distances.append((distance_of_features, datapoint))
            k_smallest_distances = self.add_to_smallest_distances(k_smallest_distances, distance_of_features, datapoint)

        # distances.sort(key=lambda tup: tup[0])
        # k_nearest = distances[:k]

        if self.classification:
            return self.process_for_classification(k_smallest_distances)
        else:
            return self.process_for_regression(k_smallest_distances)

    def test(self, dataset, query_set):
        """

        :param dataset:
        :param query_set:
        :return:
        """
        results = []
        for item in query_set:
            results.append(self.learn(dataset, item))
        return results

    # </editor-fold>

    # <editor-fold desc="Helpers">
    def add_to_smallest_distances(self, k_smallest_distances, distance_of_features, datapoint):
        """

        :param k_smallest_distances:
        :param distance_of_features:
        :param datapoint:
        :return:
        """
        if distance_of_features <= k_smallest_distances[-1][0]:
            k_smallest_distances[-1] = (distance_of_features, datapoint)
            k_smallest_distances.sort(key=lambda tup: tup[0])

        return k_smallest_distances

    def distance(self, datapoint, query):
        """
        Calculates the Euclidean distance between a datapoint in the training set and a query point.

        :param datapoint: A feature Vector that is a datapoint in the training data
        :param query:
        :return:
        """
        dist_sum = 0
        for datapoint_feature, query_feature in zip(datapoint, query):
            dist_sum += self.feature_distance(datapoint_feature, query_feature)

        dist = math.sqrt(dist_sum)

        return dist

    def feature_distance(self, datapoint_feature, query_feature):
        feature_diff = datapoint_feature - query_feature
        feature_diff *= feature_diff
        return feature_diff

    # </editor-fold>

    def process_for_regression(self, k_nearest):
        """

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

# <editor-fold desc="Validation Curves">
def generate_validation_curves(x_axis_values, values_line_1, values_line_2, label_1, label_2, x_axis_label="", y_axis_label="MSE", title=""):
    plt.plot(x_axis_values, values_line_1, '-', label=label_1)
    plt.plot(x_axis_values, values_line_2, '-', label=label_2)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    plt.legend(loc='best')

    plt.show()
    plt.close()

# </editor-fold>

# <editor-fold desc="Procedure">
def get_squared_error(predicted_value, item):
    actual_value = item[-1]
    squared_error = (predicted_value - actual_value)**2
    return squared_error


def calculate_mse(k, training_data, test_data):
    squared_error = []
    # for item in test_data:
    #     value = knn(k, training_data, item)
    #     squared_error.append(get_squared_error(value, item))
    knn = Knn(k, classification=False)
    predictions = knn.learn_many(training_data, test_data)
    for prediction, test_item in zip(predictions, test_data):
        squared_error.append(get_squared_error(prediction, test_item))

    squared_error_sum = sum(squared_error)

    mse = squared_error_sum / len(squared_error)

    return mse


def get_best_k_for_data_mse(dataset):
    k_list = range(1, 11)

    # dataset = read_file()
    random.shuffle(dataset)

    two_thirds = 2 * int(math.floor(len(dataset)/3))
    training_data = dataset[:two_thirds]
    test_data = dataset[two_thirds:]

    test_mse_list = []
    train_mse_list = []

    for k in k_list:
        test_mse_list.append(calculate_mse(k, training_data, test_data))
        train_mse_list.append(calculate_mse(k, training_data, training_data))

    generate_validation_curves(k_list, train_mse_list, test_mse_list, "Training Data Mean Squared Error",
                               "Test Data Mean Squared Error", title="Number of K's vs MSE", x_axis_label="# of k's")

    min_error = min(test_mse_list)
    return test_mse_list.index(min_error) + 1


def learning_curves(k, dataset):
    # dataset = read_file()
    random.shuffle(dataset)

    two_thirds = 2 * int(math.floor(len(dataset)/3))
    training_data = dataset[:two_thirds]
    test_data = dataset[two_thirds:]

    percent_increase = range(5, 105, 5)

    training_mse_list = []
    test_mse_list = []

    for percentage_of_data in percent_increase:
        amount_of_data = int(math.floor(len(training_data) * (percentage_of_data * .01)))
        subset_data = training_data[:amount_of_data]

        test_mse_list.append(calculate_mse(k, subset_data, test_data))
        training_mse_list.append(calculate_mse(k, subset_data, subset_data))

    generate_validation_curves(percent_increase, training_mse_list, test_mse_list, "Training Data Mean Squared Error",
                               "Test Data Mean Squared Error", title="Learning Curve for k={}".format(k),
                               x_axis_label="Percentage of data Used")

# </editor-fold>


# <editor-fold desc="Test">
# data = CustomCSVReader.read_file("data/machine.data.new.txt")
#
#
# best_k = get_best_k_for_data_mse(data)
# print(best_k)
# </editor-fold>
