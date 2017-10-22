""" Created by Max 10/4/2017 """
import sys
import pprint

from ID3Pruning import ID3Pruning
from customCsvReader import CustomCSVReader
from CrossValidation import CrossValidation


# <editor-fold desc="Experiment">
from decisionTree import ID3


def run_classification_experiment(data_set_path, learner, pruner, pruning=False, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, training and test data.

    Creates the Cross validation objects with the correct decision Tree  algorithm (classification ID3)
    Then runs the cross validation (classificaiton) and gets the outputs from the cross validation.

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with pruning = {1}".format(data_set_path, pruning))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)

    cv = CrossValidation(5, learner, pruner)
    average_error_rate = cv.cross_validation_classification(all_data, pruning)

    print("Average Error Rate: {}".format(average_error_rate[0]))
    print("Standard Deviation: {}".format(average_error_rate[1]))
    print("Average Node Count: {}".format(average_error_rate[3]))
    print("Average Node Count (Pruned): {}".format(average_error_rate[4]))
    print("Average Node Difference: {}".format(average_error_rate[2]))

    print("Last Cross Validation Set Predicted Values: \n(Predicted Value, Actual Value)")
    cv_predicted_values = average_error_rate[5]
    cv_actual_values = average_error_rate[6]
    for predicted, actual in zip(cv_predicted_values[4], cv_actual_values[4]):
        print("{0}, {1}".format(predicted, actual))

    return average_error_rate[0]
# </editor-fold>


# Decision Tree
sys.stdout = open('results/car-DT-results.txt', 'w')
run_classification_experiment("data/car.data.txt", ID3(), ID3Pruning(), pruning=False, data_type=str)
sys.stdout = open('results/car-DT-pruned-results.txt', 'w')
run_classification_experiment("data/car.data.txt", ID3(), ID3Pruning(), pruning=True, data_type=str)

sys.stdout = open('results/segmentation-DT-results.txt', 'w')
run_classification_experiment("data/segmentation.data.new.txt", ID3(), ID3Pruning(), pruning=False, data_type=float)
sys.stdout = open('results/segmentation-DT-pruned-results.txt', 'w')
run_classification_experiment("data/segmentation.data.new.txt", ID3(), ID3Pruning(), pruning=True, data_type=float)

sys.stdout = open('results/abalone-DT-results.txt', 'w')
run_classification_experiment("data/abalone.data.txt", ID3(), ID3Pruning(), pruning=False, data_type=float)
sys.stdout = open('results/abalone-DT-pruned-results.txt', 'w')
run_classification_experiment("data/abalone.data.txt", ID3(), ID3Pruning(), pruning=True, data_type=float)

