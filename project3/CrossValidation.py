""" Created by Max 10/4/2017 """
from __future__ import division
import random
import math


class CrossValidation:

    def __init__(self, folds, learner, classification=True):
        self.folds = folds
        self.learner = learner
        self.classification = classification

    def cross_validation_regression(self, dataset):
        random.shuffle(dataset)

        fold_length = int(math.floor(len(dataset)/self.folds))

        cross_validation_dataset = []
        for i in range(0, len(dataset), fold_length):
            cross_validation_dataset.append(dataset[i:i+fold_length])

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

        # print("Average MSE: {}".format(average_mse))
        # print("Standard Deviation: {}".format(sd))
        return (average_mse, sd)

    def cross_validation_classification(self, dataset):
        random.shuffle(dataset)

        fold_length = int(math.floor(len(dataset)/self.folds))

        # cross_validation_dataset = self.getStratifiedData(dataset, fold_length)
        cross_validation_dataset = []
        for i in range(0, len(dataset), fold_length):
            cross_validation_dataset.append(dataset[i:i + fold_length])

        # run cross validation
        error_list = []
        for i in range(self.folds):
            # construct training set.
            test_set = cross_validation_dataset[i]
            training_set = cross_validation_dataset[:i] + cross_validation_dataset[i+1:]
            training_set = [item for sublist in training_set for item in sublist]

            error_list.append(self.calculate_error_rate(self.learner, training_set, test_set))

        average_error_rate = sum(error_list) / len(error_list)
        sd = self.calc_standard_deviation(average_error_rate, error_list)

        # print("Average Error Rate: {}".format(average_error_rate))
        # print("Standard Deviation: {}".format(sd))
        return (average_error_rate, sd)



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

    def calculate_error_rate(self, learner, training_data, test_data):
        predictions = learner.test(training_data, test_data)

        num_errors = 0
        for prediction, test_item in zip(predictions, test_data):
            if prediction != test_item[-1]:
                num_errors += 1

        error_rate = num_errors / len(predictions)
        return error_rate


    def getStratifiedData(self, dataset, fold_length):
        unique_labels = {}
        labeled_datapoints = {}
        for datapoint in dataset:
            label = datapoint[-1]
            if label in unique_labels:
                unique_labels[label] += 1
                labeled_datapoints[label].append(datapoint)
            else:
                unique_labels[label] = 1
                labeled_datapoints[label] = [datapoint]

        print(unique_labels)
        print(len(unique_labels))
        print(labeled_datapoints)

