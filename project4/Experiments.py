""" Created by Max 10/4/2017 """
import sys
import pprint

from customCsvReader import CustomCSVReader
from CrossValidation import CrossValidation


# <editor-fold desc="Experiment">
def run_experiment_regression(data_set_path, k_nearest, learner, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, trainnig and test data.

    Creates the Cross validation objects with the correct k-NN algorithm (classification k-NN)
    Then runs the cross validation and gets the outputs from the cross validation

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with k nearest = {1}".format(data_set_path, k_nearest))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)

    # For algorithms knn Cross Validation
    cv = CrossValidation(5, learner)
    average_mse = cv.cross_validation_regression(all_data)

    print("Average MSE: {}".format(average_mse[0]))
    print("Standard Deviation: {}".format(average_mse[1]))

    print("Last Cross Validation Set Predicted Values: \n(Predicted Value, Actual Value)")
    cv_predicted_values = average_mse[2]
    cv_actual_values = average_mse[3]
    for predicted, actual in zip(cv_predicted_values[4], cv_actual_values[4]):
        print("{0}, {1}".format(predicted, actual))

    return average_mse[0]


def run_classification_experiment(data_set_path, k_nearest, learner, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, training and test data.

    Creates the Cross validation objects with the correct decision Tree  algorithm (classification ID3)
    Then runs the cross validation (classificaiton) and gets the outputs from the cross validation.

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with k nearest = {1}".format(data_set_path, k_nearest))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)

    cv = CrossValidation(5, learner)
    average_error_rate = cv.cross_validation_classification(all_data)

    print("Average Error Rate: {}".format(average_error_rate[0]))
    print("Standard Deviation: {}".format(average_error_rate[1]))

    print("Last Cross Validation Set Predicted Values: \n(Predicted Value, Actual Value)")
    cv_predicted_values = average_error_rate[2]
    cv_actual_values = average_error_rate[3]
    for predicted, actual in zip(cv_predicted_values[4], cv_actual_values[4]):
        print("{0}, {1}".format(predicted, actual))

    return average_error_rate[0]

# </editor-fold>



# KNN & Condensed KNN experiments
# sys.stdout = open('results/knn-machine-results.txt', 'w')
# run_experiment_regression("data/machine.data.new.txt", 2, Knn(2))

