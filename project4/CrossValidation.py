""" Created by Max 10/4/2017 """
from __future__ import division
import random
import math

from typing import Dict, Tuple, List


class CrossValidation:

    def __init__(self, folds, learner, pruner):
        """
        Constructor
        :param folds: num folds
        :param learner: the k-NN algorithm to use.
        """
        self.folds = folds
        self.learner = learner
        self.pruner = pruner

    def cross_validation_classification(self, dataset, pruning=False):
        """
        Runs cross validation, using the k-NN classification

        Creates the folds for CV. USES Stratified data for each fold

        For each fold, creates the test and training sets.
        Calculate the error rate for the sets.
        Store the results.

        average the error rate over the number of folds and calc SD

        return values.

        :param dataset: the training data set to use.
        :param pruning: boolean, if we are using pruning or not.
        :return: average MSE, Standard deviation, the predictions, and the actuals for all cv runs
        """
        random.shuffle(dataset)

        validation_set, remaining_data_set = self.get_validation_set(dataset, 10)
        dataset = remaining_data_set

        fold_length = int(math.floor(len(dataset)/self.folds))

        cross_validation_dataset = self.get_stratified_data(dataset, fold_length, self.folds)

        # run cross validation
        error_list = []
        predictions = []
        actuals = []
        difference_in_node_number = []
        model_node_number = []
        pruned_model_node_number = []
        for i in range(self.folds):
            # construct training set.
            test_set = cross_validation_dataset[i]
            training_set = cross_validation_dataset[:i] + cross_validation_dataset[i+1:]
            training_set = [item for sublist in training_set for item in sublist]

            # calculate the error rate for the test set with the training set
            model = self.learner.learn(training_set)
            model_node_count = self.learner.node_count(model)
            model_node_number.append(model_node_count)
            # If using pruning, prune the model
            if pruning:
                pruned_model = self.pruner.prune(model, validation_set)
                pruned_model_node_count = self.learner.node_count(pruned_model)
                difference_in_node_number.append(model_node_count - pruned_model_node_count)
                pruned_model_node_number.append(pruned_model_node_count)
                model = pruned_model

            error_rate = self.calculate_error_rate(self.learner, model, test_set)

            # Store results
            error_list.append(error_rate[0])
            predictions.append(error_rate[1])
            actuals.append(error_rate[2])

        average_error_rate = sum(error_list) / len(error_list)
        sd = self.calc_standard_deviation(average_error_rate, error_list)

        average_model_node_number = sum(model_node_number) / len(model_node_number)

        if pruning:
            average_node_count_difference = sum(difference_in_node_number) / len(difference_in_node_number)
            average_pruned_model_node_number = sum(pruned_model_node_number) / len(pruned_model_node_number)
        else:
            average_pruned_model_node_number = "N/A"
            average_node_count_difference = "N/A"

        # print("Average Error Rate: {}".format(average_error_rate))
        # print("Standard Deviation: {}".format(sd))
        return average_error_rate, sd, average_node_count_difference, average_model_node_number,\
               average_pruned_model_node_number, predictions, actuals


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

    def calculate_error_rate(self, learner, model, test_data):
        """
        Calculates the error rate for classification.

        tracks the actual and predictions

        :param learner: k-NN classifier
        :param training_data: model of data for k-NN
        :param test_data: query points for k-NN
        :return: error_rate, list of predictions, the actual values.
        """
        predictions = learner.classify(model, test_data)
        actuals = []

        num_errors = 0
        for prediction, test_item in zip(predictions, test_data):
            actuals.append(test_item[-1])
            if prediction != test_item[-1]:
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

    def get_validation_set(self, dataset, percentage_of_data_for_validation) -> Tuple[List[list], List[list]]:
        fold_length = int(math.floor(len(dataset) / percentage_of_data_for_validation))

        stratified_data = self.get_stratified_data(dataset, fold_length, 1)
        stratified_data = stratified_data[0]

        for stratified_data_point in stratified_data:
            if stratified_data_point in dataset:
                index = dataset.index(stratified_data_point)
                del dataset[index]

        return stratified_data, dataset
