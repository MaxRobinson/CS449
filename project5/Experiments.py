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
def run_classification_experiment(data_set_path, learner, positive_class_name, linearRegression=False, data_type=float):
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

    if not linearRegression:
        print("Learned Naive Bayes Distribution: ")
        print("Keys are structured as follows: (feature#, possible domain values 0 or 1, 'label', label value)")
        print("Special Key's that are ('label', possible_class_value) are the percentage of the distribution with "
              "that class label")
    else:
        print("Learned Linear Regression Model (Thetas) ")

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(average_error_rate[2][4])
    print()

    print("Last Cross Validation Set Predicted Values: \n(Predicted Value, Actual Value)")
    cv_predicted_values = average_error_rate[3]
    cv_actual_values = average_error_rate[4]
    for predicted, actual in zip(cv_predicted_values[4], cv_actual_values[4]):
        if linearRegression:
            print("{0}, {1}".format(predicted[0], actual))
        else:
            print("{0}, {1}".format(predicted, actual))

    return average_error_rate[0]
# </editor-fold>

# Naive Bayes
sys.stdout = open('results/NB-Breast-Cancer-results-class-1.txt', 'w')
run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", NaiveBayes(), 1)

sys.stdout = open('results/NB-Breast-Cancer-results-class-2.txt', 'w')
run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", NaiveBayes(), 0)

sys.stdout = open('results/NB-soybean-small-results-class-D1.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", NaiveBayes(), "D1")
sys.stdout = open('results/NB-soybean-small-results-class-D2.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", NaiveBayes(), "D2")
sys.stdout = open('results/NB-soybean-small-results-class-D3.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", NaiveBayes(), "D3")
sys.stdout = open('results/NB-soybean-small-results-class-D4.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", NaiveBayes(), "D4")

sys.stdout = open('results/NB-house-votes-84-results-class-democrat.txt', 'w')
run_classification_experiment("data/house-votes-84.data.new.txt", NaiveBayes(), "democrat")
sys.stdout = open('results/NB-house-votes-84-results-class-republican.txt', 'w')
run_classification_experiment("data/house-votes-84.data.new.txt", NaiveBayes(), "republican")

sys.stdout = open('results/NB-iris-results-class-setosa.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", NaiveBayes(), "Iris-setosa")
sys.stdout = open('results/NB-iris-results-class-versicolor.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", NaiveBayes(), "Iris-versicolor")
sys.stdout = open('results/NB-iris-results-class-virginica.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", NaiveBayes(), "Iris-virginica")

sys.stdout = open('results/NB-glass-results-class-1.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 1)
sys.stdout = open('results/NB-glass-results-class-2.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 2)
sys.stdout = open('results/NB-glass-results-class-3.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 3)
sys.stdout = open('results/NB-glass-results-class-5.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 5)
sys.stdout = open('results/NB-glass-results-class-6.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 6)
sys.stdout = open('results/NB-glass-results-class-7.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NaiveBayes(), 7)
#
# # Logistic regression
sys.stdout = open('results/LR-Bresat-Cancer-results-class-1.txt', 'w')
run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", LogisticRegression(), 1, linearRegression=True)
sys.stdout = open('results/LR-Bresat-Cancer-results-class-2.txt', 'w')
run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", LogisticRegression(), 0, linearRegression=True)

sys.stdout = open('results/LR-soybean-small-results-class-D1.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", LogisticRegression(), "D1", linearRegression=True)
sys.stdout = open('results/LR-soybean-small-results-class-D2.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", LogisticRegression(), "D2", linearRegression=True)
sys.stdout = open('results/LR-soybean-small-results-class-D3.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", LogisticRegression(), "D3", linearRegression=True)
sys.stdout = open('results/LR-soybean-small-results-class-D4.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", LogisticRegression(), "D4", linearRegression=True)

sys.stdout = open('results/LR-house-votes-84-results-class-democrat.txt', 'w')
run_classification_experiment("data/house-votes-84.data.new.txt", LogisticRegression(), "democrat", linearRegression=True)
sys.stdout = open('results/LR-house-votes-84-results-class-republican.txt', 'w')
run_classification_experiment("data/house-votes-84.data.new.txt", LogisticRegression(), "republican", linearRegression=True)

sys.stdout = open('results/LR-iris-results-class-setosa.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", LogisticRegression(), "Iris-setosa", linearRegression=True)
sys.stdout = open('results/LR-iris-results-class-versicolor.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", LogisticRegression(), "Iris-versicolor", linearRegression=True)
sys.stdout = open('results/LR-iris-results-class-virginica.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", LogisticRegression(), "Iris-virginica", linearRegression=True)

sys.stdout = open('results/LR-glass-results-class-1.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", LogisticRegression(), 1, linearRegression=True)
sys.stdout = open('results/LR-glass-results-class-2.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", LogisticRegression(), 2, linearRegression=True)
sys.stdout = open('results/LR-glass-results-class-3.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", LogisticRegression(), 3, linearRegression=True)
sys.stdout = open('results/LR-glass-results-class-5.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", LogisticRegression(), 5, linearRegression=True)
sys.stdout = open('results/LR-glass-results-class-6.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", LogisticRegression(), 6, linearRegression=True)
sys.stdout = open('results/LR-glass-results-class-7.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", LogisticRegression(), 7, linearRegression=True)
