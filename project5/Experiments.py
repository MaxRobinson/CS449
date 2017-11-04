""" Created by Max 10/4/2017 """
import sys
import pprint

from LogisticRegression import LogisticRegression
from NaiveBayes import NaiveBayes
from customCsvReader import CustomCSVReader
from CrossValidation import CrossValidation


# <editor-fold desc="Experiment">
from decisionTree import ID3


# def run_classification_experiment(data_set_path, learner, pruner, pruning=False, data_type=float):
def run_classification_experiment(data_set_path, learner,  positive_class_name, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, training and test data.

    Creates the Cross validation objects with the correct decision Tree  algorithm (classification ID3)
    Then runs the cross validation (classificaiton) and gets the outputs from the cross validation.

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with positive class = {1}".format(data_set_path, positive_class_name))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)

    # Pre-process the data to split into 2 classes, positive and not positive.
    all_data = learner.pre_process(all_data, positive_class_name)

    cv = CrossValidation(5, learner)
    average_error_rate = cv.cross_validation_classification(all_data)

    print("Average Error Rate: {}".format(average_error_rate[0]))
    print("Standard Deviation: {}".format(average_error_rate[1]))

    print("Learned Naive Bayes Distribution: ")
    print("Keys are structured as follows: (feature#, possible domain values 0 or 1, 'label', label value)")
    print("Special Key's that are ('label', possible_class_value) are the percentage of the distribution with "
          "that class label")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(average_error_rate[2][4])
    print()

    print("Last Cross Validation Set Predicted Values: \n(Predicted Value, Actual Value)")
    cv_predicted_values = average_error_rate[3]
    cv_actual_values = average_error_rate[4]
    for predicted, actual in zip(cv_predicted_values[4], cv_actual_values[4]):
        print("{0}, {1}".format(predicted, actual))

    return average_error_rate[0]
# </editor-fold>

# Naive Bayes
# sys.stdout = open('results/NB-Bresat-Cancer-results.txt', 'w')
# run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", NaiveBayes(), 1)
# run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", NaiveBayes(), 0)
#
# # sys.stdout = open('results/NB-soybean-small-results.txt', 'w')
# run_classification_experiment("data/soybean-small.data.new.txt", NaiveBayes(), "D1")
# run_classification_experiment("data/soybean-small.data.new.txt", NaiveBayes(), "D2")
# run_classification_experiment("data/soybean-small.data.new.txt", NaiveBayes(), "D3")
# run_classification_experiment("data/soybean-small.data.new.txt", NaiveBayes(), "D4")
#
# # sys.stdout = open('results/NB-house-votes-84-results.txt', 'w')
# run_classification_experiment("data/house-votes-84.data.new.txt", NaiveBayes(), "democrat")
# run_classification_experiment("data/house-votes-84.data.new.txt", NaiveBayes(), "republican")
#
# # sys.stdout = open('results/NB-iris-results.txt', 'w')
# run_classification_experiment("data/iris.data.new.txt", NaiveBayes(), "Iris-setosa")
# run_classification_experiment("data/iris.data.new.txt", NaiveBayes(), "Iris-versicolor")
# run_classification_experiment("data/iris.data.new.txt", NaiveBayes(), "Iris-virginica")
#
# # sys.stdout = open('results/NB-glass-results.txt', 'w')
# run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 1)
# run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 2)
# run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 3)
# run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 5)
# run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 6)
# run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 7)

# Logistic regression
run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", LogisticRegression(), 1)
# run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", LogisticRegression(), 0)
