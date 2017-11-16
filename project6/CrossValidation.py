""" Created by Max 10/4/2017 """
from __future__ import division
import random
import math

from typing import Dict, Tuple, List


class CrossValidation:

    def __init__(self, folds, learner):
        """
        Constructor
        :param folds: num folds
        :param learner: the k-NN algorithm to use.
        """
        self.folds = folds
        self.learner = learner

    def cross_validation_regression(self, dataset):
        """
        Runs cross validation, using the k-NN regression

        Creates the folds for CV.
        For each fold, creates the test and training sets.
        Calculate the MSE for the data sets.
        Store the results.

        average the MSE over the number of folds and calc SD

        return values.

        :param dataset: the training data set to use.
        :return: average MSE, Standard deviation, the predictions, and the actuals for all cv runs
        """
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

            # Get the MSE
            mse = self.calculate_mse(self.learner, training_set, test_set)

            # Store results
            mse_list.append(mse[0])
            predictions.append(mse[1])
            actuals.append(mse[2])

        average_mse = sum(mse_list) / len(mse_list)
        sd = self.calc_standard_deviation(average_mse, mse_list)

        return average_mse, sd, predictions, actuals

    def cross_validation_classification(self, dataset, pruning=False):
        """
        Runs cross validation, using the DT classification

        Pulls out a stratified sample for the validation set. 10%

        Creates the folds for CV. USES Stratified data for each fold

        For each fold, creates the test and training sets.
        Calculate the error rate for the sets.
        Store the results.

        average the error rate over the number of folds and calc SD

        return values.

        :param dataset: the training data set to use.
        :param pruning: boolean, if we are using pruning or not.
        :return: average Error rate , Standard deviation, the predictions, and the actuals for all cv runs
        """
        random.shuffle(dataset)

        fold_length = int(math.floor(len(dataset)/self.folds))

        cross_validation_dataset = self.get_stratified_data(dataset, fold_length, self.folds)

        # run cross validation
        error_list = []
        predictions = []
        actuals = []
        models = []
        for i in range(self.folds):
            # construct training set.
            test_set = cross_validation_dataset[i]
            training_set = cross_validation_dataset[:i] + cross_validation_dataset[i+1:]
            training_set = [item for sublist in training_set for item in sublist]

            # calculate the error rate for the test set with the training set
            model = self.learner.learn(training_set)
            models.append(model)

            error_rate = self.calculate_error_rate(self.learner, model, test_set)

            # Store results
            error_list.append(error_rate[0])
            predictions.append(error_rate[1])
            actuals.append(error_rate[2])

        average_error_rate = sum(error_list) / len(error_list)
        sd = self.calc_standard_deviation(average_error_rate, error_list)

        return average_error_rate, sd, models, predictions, actuals


    def calc_standard_deviation(self, average, list_of_values):
        """
        Calculates the SD of the Cross validation.
        :param average: average error for CV float
        :param list_of_values: list of errors for CV
        :return: sd of CV
        """
        sd = 0
        for x in list_of_values:
            sd += (x - average) ** 2

        sd /= len(list_of_values)
        sd = math.sqrt(sd)

        return sd

    def calculate_mse(self, learner, training_data, test_data):
        """
        Helper function for calculating MSE, and tracking actual and predicted values.

        Calculate the squared error for each pair of points, and sum over all the squared errors.
        Then divide by the number of squared errors.

        :param learner: the k-NN regression class
        :param training_data: the data set for the model
        :param test_data: the query points to get predictions for.
        :return: (MSE, list of predictions, list of corresponding actuals)
        """
        squared_error = []
        predictions = learner.test(training_data, test_data)
        actual = []

        for prediction, test_item in zip(predictions, test_data):
            actual.append(test_item[-1])
            squared_error.append(self.get_squared_error(prediction, test_item))

        squared_error_sum = sum(squared_error)

        mse = squared_error_sum / len(squared_error)

        return mse, predictions, actual

    def get_squared_error(self, predicted_value, item):
        """
        Calculates the squared error of predicted points, and actual items

        :param predicted_value: list of floats
        :param item: list of query data points.
        :return: Squared error
        """
        actual_value = item[-1]
        squared_error = (predicted_value - actual_value)**2
        return squared_error

    def calculate_error_rate(self, learner, model, test_data):
        """
        Calculates the error rate for classification.

        tracks the actual and predictions

        :param learner: a classifier
        :param model: model of the learner
        :param test_data: query points for
        :return: error_rate, list of predictions, the actual values.
        """
        predictions = learner.classify(model, test_data)
        actuals = []

        num_errors = 0
        for prediction, test_item in zip(predictions, test_data):
            if type(prediction) is tuple:
                actual_prediction = prediction[0]
            else:
                actual_prediction = prediction[0][0]

            actuals.append(test_item[-1])
            if actual_prediction != test_item[-1]:
                num_errors += 1

        error_rate = num_errors / len(predictions)
        return (error_rate, predictions, actuals)

    def get_stratified_data(self, dataset, fold_length, num_folds):
        """
        Creates the Cross Validation fold with Stratified data, i.e. data that matches the distribution of the overall
        data set.

        segment the data
        calculate distribution of classes
        create x folds according to that distribution, without replacement.


        :param dataset: list of list of data points with labels
        :param fold_length: number of points in each fold
        :param num_folds: number of folds to build.
        :return: a list of list of datapoints, where each inner list is a fold in the CV
        """
        # for all data
        unique_labels = {}
        labeled_datapoints = {}
        # build dict of listed segmented datapoints
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
        """
        Builds a single CV fold according to a distribution without replacement.

        :param distribution: dict of dist of the classes
        :param labeled_datapoints: dict of labeled data points
        :param fold_length: number of data poitns in a fold
        :return: list of data points in the fold. 
        """
        # build a single fold
        fold = []
        for key in distribution:
            # get number of data points for this class
            number_of_items_per_class = int(distribution[key] * fold_length)
            if number_of_items_per_class == 0:
                number_of_items_per_class = 1

            # select the data points for this class in this fold.
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

    def get_validation_set(self, dataset: List[List], percentage_of_data_for_validation: int) -> Tuple[List[list], List[list]]:
        """
        Creates a validation set of the data and deletes the data used for validation from the original data set

        :param dataset: list of lsit
        :param percentage_of_data_for_validation: int percentage to use
        :return: tuple( validation set, modified data set
        """
        fold_length = int(math.floor(len(dataset) / percentage_of_data_for_validation))

        stratified_data = self.get_stratified_data(dataset, fold_length, 1)
        stratified_data = stratified_data[0]

        for stratified_data_point in stratified_data:
            if stratified_data_point in dataset:
                index = dataset.index(stratified_data_point)
                del dataset[index]

        return stratified_data, dataset
