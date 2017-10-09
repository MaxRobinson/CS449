""" Created by Max 10/4/2017 """
import sys
import pprint

from customCsvReader import CustomCSVReader
from CrossValidation import CrossValidation
from knn import Knn
from condensedNN import CondensedNN


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

    Works by reading in the data, trainnig and test data.

    Creates the Cross validation objects with the correct k-NN algorithm (classification k-NN)
    Then runs the cross validation (classificaiton) and gets the outputs from the cross validation

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


def run_classification_experiment_condensed(data_set_path, k_nearest, learner, condenser, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, trainnig and test data.

    Creates the Cross validation objects with the correct k-NN algorithm (condensed k-NN)
    Then runs the cross validation (condensed) and gets the outputs from the cross validation

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with k nearest = {1}".format(data_set_path, k_nearest))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)

    cv = CrossValidation(5, learner)
    average_error_rate = cv.cross_validation_classification_condensed(condenser, all_data)

    print("Average Error Rate: {}".format(average_error_rate[0]))
    print("Standard Deviation: {}".format(average_error_rate[1]))
    print()

    print("Selected Condensed Data Points for Last Cross Validation:")
    pp = pprint.PrettyPrinter(indent=2, width=400)
    pp.pprint(average_error_rate[2][4])
    print()

    print("Last Cross Validation Set Predicted Values: \n(Predicted Value, Actual Value)")
    cv_predicted_values = average_error_rate[3]
    cv_actual_values = average_error_rate[4]
    for predicted, actual in zip(cv_predicted_values[4], cv_actual_values[4]):
        print("{0}, {1}".format(predicted, actual))

    return average_error_rate[0]

# </editor-fold>



# KNN & Condensed KNN experiments
sys.stdout = open('results/knn-machine-results.txt', 'w')
run_experiment_regression("data/machine.data.new.txt", 2, Knn(2))

sys.stdout = open('results/knn-forestfires-results.txt', 'w')
run_experiment_regression("data/forestfires.data.new.txt", 10, Knn(10))


# Classification
sys.stdout = open('results/knn-ecoli-results.txt', 'w')
run_classification_experiment("data/ecoli.data.new.txt", 8, Knn(8))

sys.stdout = open('results/knn-segmentation-results.txt', 'w')
run_classification_experiment("data/segmentation.data.new.txt", 1, Knn(1))

# Condensed KNN
sys.stdout = open('results/condensed-knn-ecoli-results.txt', 'w')
run_classification_experiment_condensed("data/ecoli.data.new.txt", 6, Knn(6), CondensedNN())

sys.stdout = open('results/condensed-knn-segmentation-results.txt', 'w')
run_classification_experiment_condensed("data/segmentation.data.new.txt", 1, Knn(1), CondensedNN())
