""" Created by Max 10/4/2017 """
from __future__ import division
import random
import math


class CrossValidation:

    def __init__(self, folds, learner, classification=True):
        self.folds = folds
        self.learner = learner
        self.classification = classification

    def cross_validation(self, dataset):
        random.shuffle(dataset)

        tenth_length = int(math.floor(len(dataset)/self.folds))

        cross_validation_dataset = []
        for i in range(0, len(dataset), tenth_length):
            cross_validation_dataset.append(dataset[i:i+tenth_length])

        # run cross validation
        mse_list = []
        for i in range(self.folds):
            # construct training set.
            test_set = cross_validation_dataset[i]
            training_set = cross_validation_dataset[:i] + cross_validation_dataset[i+1:]
            training_set = [item for sublist in training_set for item in sublist]

            mse_list.append(self.calculate_mse(self.learner, training_set, test_set))

        average_mse = sum(mse_list) / len(mse_list)
        sd = self.calc_standard_deviation(average_mse, mse_list)

        print("Average MSE: {}".format(average_mse))
        print("Standard Deviation: {}".format(sd))

    def calculate_mse(self, learner, training_data, test_data):
        squared_error = []
        # for item in test_data:
        #     value = knn(k, training_data, item)
        #     squared_error.append(get_squared_error(value, item))
        predictions = learner.test(training_data, test_data)
        for prediction, test_item in zip(predictions, test_data):
            squared_error.append(self.get_squared_error(prediction, test_item))

        squared_error_sum = sum(squared_error)

        mse = squared_error_sum / len(squared_error)

        return mse

    def get_squared_error(self, predicted_value, item):
        actual_value = item[-1]
        squared_error = (predicted_value - actual_value)**2
        return squared_error

    def calc_standard_deviation(self, average, list_of_values):
        sd = 0
        for x in list_of_values:
            sd += (x - average) ** 2

        sd /= len(list_of_values)
        sd = math.sqrt(sd)

        return sd