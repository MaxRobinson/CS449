""" Created by Max 10/4/2017 """
from __future__ import division
import random
import math


class CrossValidation:

    def __init__(self, folds, learner):
        self.folds = folds
        self.learner = learner

    def cross_validation_regression(self, dataset):
        random.shuffle(dataset)

        fold_length = int(math.floor(len(dataset)/self.folds))

        cross_validation_dataset = []
        for i in range(0, len(dataset), fold_length):
            cross_validation_dataset.append(dataset[i:i+fold_length])

        # run cross validation
        mse_list = []
        predictions = []
        actuals = []
        for i in range(self.folds):
            # construct training set.
            test_set = cross_validation_dataset[i]
            training_set = cross_validation_dataset[:i] + cross_validation_dataset[i+1:]
            training_set = [item for sublist in training_set for item in sublist]

            # mse_list.append(self.calculate_mse(self.learner, training_set, test_set))
            mse = self.calculate_mse(self.learner, training_set, test_set)
            mse_list.append(mse[0])
            predictions.append(mse[1])
            actuals.append(mse[2])

        average_mse = sum(mse_list) / len(mse_list)
        sd = self.calc_standard_deviation(average_mse, mse_list)

        # print("Average MSE: {}".format(average_mse))
        # print("Standard Deviation: {}".format(sd))
        return (average_mse, sd, predictions, actuals)

    def cross_validation_classification(self, dataset):
        random.shuffle(dataset)

        fold_length = int(math.floor(len(dataset)/self.folds))

        cross_validation_dataset = self.get_stratified_data(dataset, fold_length, self.folds)
        # cross_validation_dataset = []
        # for i in range(0, len(dataset), fold_length):
        #     cross_validation_dataset.append(dataset[i:i + fold_length])

        # run cross validation
        error_list = []
        predictions = []
        actuals = []
        for i in range(self.folds):
            # construct training set.
            test_set = cross_validation_dataset[i]
            training_set = cross_validation_dataset[:i] + cross_validation_dataset[i+1:]
            training_set = [item for sublist in training_set for item in sublist]

            # calculate the error rate for the test set with the training set
            error_rate = self.calculate_error_rate(self.learner, training_set, test_set)
            error_list.append(error_rate[0])
            predictions.append(error_rate[1])
            actuals.append(error_rate[2])

        average_error_rate = sum(error_list) / len(error_list)
        sd = self.calc_standard_deviation(average_error_rate, error_list)

        # print("Average Error Rate: {}".format(average_error_rate))
        # print("Standard Deviation: {}".format(sd))
        return (average_error_rate, sd, predictions, actuals)

    def cross_validation_classification_condensed(self, condenser, dataset):
        random.shuffle(dataset)

        fold_length = int(math.floor(len(dataset) / self.folds))

        # cross_validation_dataset = self.getStratifiedData(dataset, fold_length)
        cross_validation_dataset = []
        for i in range(0, len(dataset), fold_length):
            cross_validation_dataset.append(dataset[i:i + fold_length])

        # run cross validation
        error_list = []
        predictions = []
        actuals = []
        condensed_datasets = []
        for i in range(self.folds):
            # construct training set.
            test_set = cross_validation_dataset[i]
            training_set = cross_validation_dataset[:i] + cross_validation_dataset[i + 1:]
            training_set = [item for sublist in training_set for item in sublist]

            # Condense Data of training set
            condensed_dataset = condenser.condense(training_set)
            condensed_datasets.append(condensed_dataset)

            # calculate the error rate on the condensed set
            error_rate = self.calculate_error_rate(self.learner, condensed_dataset, test_set)
            error_list.append(error_rate[0])
            predictions.append(error_rate[1])
            actuals.append(error_rate[2])

        average_error_rate = sum(error_list) / len(error_list)
        sd = self.calc_standard_deviation(average_error_rate, error_list)

        return (average_error_rate, sd, condensed_datasets, predictions, actuals)


    def calculate_mse(self, learner, training_data, test_data):
        squared_error = []
        # for item in test_data:
        #     value = knn(k, training_data, item)
        #     squared_error.append(get_squared_error(value, item))
        predictions = learner.test(training_data, test_data)
        actual = []

        for prediction, test_item in zip(predictions, test_data):
            actual.append(test_item[-1])
            squared_error.append(self.get_squared_error(prediction, test_item))

        squared_error_sum = sum(squared_error)

        mse = squared_error_sum / len(squared_error)

        return mse, predictions, actual

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
        actuals = []

        num_errors = 0
        for prediction, test_item in zip(predictions, test_data):
            actuals.append(test_item[-1])
            if prediction != test_item[-1]:
                num_errors += 1

        error_rate = num_errors / len(predictions)
        return (error_rate, predictions, actuals)


    def get_stratified_data(self, dataset, fold_length, num_folds):
        # for all data
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

        # Calculate the class distribution
        distribution = {}
        for key in unique_labels:
            distribution[key] = unique_labels[key] / len(dataset)

        fold_data_set = []
        for x in range(num_folds):
            single_fold = self.build_single_fold(distribution, labeled_datapoints, fold_length)
            fold_data_set.append(single_fold)

        return fold_data_set




    def build_single_fold(self, distribution, labeled_datapoints, fold_length):
        # build a single fold
        fold = []
        for key in distribution:
            number_of_items_per_class = int(distribution[key] * fold_length)
            if number_of_items_per_class == 0:
                number_of_items_per_class = 1

            single_key_datapoints = []
            for x in range(number_of_items_per_class):
                datapoint_possibilities = labeled_datapoints[key]
                if len(datapoint_possibilities) == 0:
                    continue

                selected_datapoint_index = random.randint(0, len(datapoint_possibilities) - 1)
                single_key_datapoints.append(datapoint_possibilities[selected_datapoint_index])

                # remove datapoint from being selected again.
                del datapoint_possibilities[selected_datapoint_index]

            fold = fold + single_key_datapoints

        return fold

