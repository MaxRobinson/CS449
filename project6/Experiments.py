""" Created by Max 10/4/2017 """
import sys
import pprint

from NeuralNetworkFast import NeuralNetwork
from customCsvReader import CustomCSVReader
from CrossValidation import CrossValidation


# <editor-fold desc="Experiment">
def run_classification_experiment(data_set_path, learner, positive_class_name, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, training and test data.

    Creates the Cross validation objects with the correct linear model algorithm (NB or LR)
    Then runs the cross validation (classificaiton) and gets the outputs from the cross validation.

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with positive class = {1}".format(data_set_path, positive_class_name))

    # Network structure.
    print("Number of Hidden Layers: {}".format(len(learner.weights)-1))
    print("Number of Nodes in First Hidden Layer: {}".format(learner.num_in_hidden_layer_1))
    print("Number of Nodes in First Hidden Layer: {}".format(learner.num_in_hidden_layer_2))

    all_data = CustomCSVReader.read_file(data_set_path, data_type)

    # Pre-process the data to split into 2 classes, positive and not positive.
    all_data = learner.pre_process(all_data, positive_class_name)

    cv = CrossValidation(5, learner)
    average_error_rate = cv.cross_validation_classification(all_data)

    print("Average Error Rate: {}".format(average_error_rate[0]))
    print("Standard Deviation: {}".format(average_error_rate[1]))

    # if not linearRegression:
    #     print("Learned Naive Bayes Distribution: ")
    #     print("Keys are structured as follows: (feature#, possible domain values 0 or 1, 'label', label value)")
    #     print("Special Key's that are ('label', possible_class_value) are the percentage of the distribution with "
    #           "that class label")
    # else:
    print("Learned NN Model (Weights) ")

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(average_error_rate[2][4])
    print()

    print("Last Cross Validation Set Predicted Values: \n(Predicted Value, Actual Value)")
    cv_predicted_values = average_error_rate[3]
    cv_actual_values = average_error_rate[4]
    for predicted, actual in zip(cv_predicted_values[4], cv_actual_values[4]):
        # if linearRegression:
        #     print("{0}, {1}".format(predicted[0], actual))
        # else:
        print("{0}, {1}".format(predicted, actual))

    return average_error_rate[0]


# </editor-fold>
# NeuralNetwork 0 hidden layers
# sys.stdout = open('results/NN-0-hidden-layers-Cancer-results-class-1.txt', 'w')
# run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", NeuralNetwork(num_inputs=90, num_outputs=1), 1)
#
# sys.stdout = open('results/NN-0-hidden-layers-Cancer-results-class-0.txt', 'w')
# run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", NeuralNetwork(num_inputs=90, num_outputs=1), 0)
#
# sys.stdout = open('results/NN-0-hidden-layers-soybean-small-results-class-D1.txt', 'w')
# run_classification_experiment("data/soybean-small.data.new.txt", NeuralNetwork(num_inputs=204, num_outputs=1), "D1")
# sys.stdout = open('results/NN-0-hidden-layers-soybean-small-results-class-D2.txt', 'w')
# run_classification_experiment("data/soybean-small.data.new.txt", NeuralNetwork(num_inputs=204, num_outputs=1), "D2")
# sys.stdout = open('results/NN-0-hidden-layers-soybean-small-results-class-D3.txt', 'w')
# run_classification_experiment("data/soybean-small.data.new.txt", NeuralNetwork(num_inputs=204, num_outputs=1), "D3")
# sys.stdout = open('results/NN-0-hidden-layers-soybean-small-results-class-D4.txt', 'w')
# run_classification_experiment("data/soybean-small.data.new.txt", NeuralNetwork(num_inputs=204, num_outputs=1), "D4")
#
# sys.stdout = open('results/NN-0-hidden-layers-house-votes-84-results-class-democrat.txt', 'w')
# run_classification_experiment("data/house-votes-84.data.new.txt", NeuralNetwork(num_inputs=16, num_outputs=1), "democrat")
# sys.stdout = open('results/NN-0-hidden-layers-house-votes-84-results-class-republican.txt', 'w')
# run_classification_experiment("data/house-votes-84.data.new.txt", NeuralNetwork(num_inputs=16, num_outputs=1), "republican")
#
# sys.stdout = open('results/NN-0-hidden-layers-iris-results-class-setosa.txt', 'w')
# run_classification_experiment("data/iris.data.new.txt", NeuralNetwork(num_inputs=24, num_outputs=1), "Iris-setosa")
# sys.stdout = open('results/NN-0-hidden-layers-iris-results-class-versicolor.txt', 'w')
# run_classification_experiment("data/iris.data.new.txt", NeuralNetwork(num_inputs=24, num_outputs=1), "Iris-versicolor")
# sys.stdout = open('results/NN-0-hidden-layers-iris-results-class-virginica.txt', 'w')
# run_classification_experiment("data/iris.data.new.txt", NeuralNetwork(num_inputs=24, num_outputs=1), "Iris-virginica")
#
# sys.stdout = open('results/NN-0-hidden-layers-glass-results-class-1.txt', 'w')
# run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1), 1)
# sys.stdout = open('results/NN-0-hidden-layers-glass-results-class-2.txt', 'w')
# run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1), 2)
# sys.stdout = open('results/NN-0-hidden-layers-glass-results-class-3.txt', 'w')
# run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1), 3)
# sys.stdout = open('results/NN-0-hidden-layers-glass-results-class-5.txt', 'w')
# run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1), 5)
# sys.stdout = open('results/NN-0-hidden-layers-glass-results-class-6.txt', 'w')
# run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1), 6)
# sys.stdout = open('results/NN-0-hidden-layers-glass-results-class-7.txt', 'w')
# run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1), 7)

console_out = sys.stdout

# NeuralNetwork 1 hidden layers

sys.stdout = console_out
print("Cancer Data Set")

sys.stdout = open('results/NN-1-hidden-layers-Cancer-results-class-1.txt', 'w')
run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", NeuralNetwork(num_inputs=90, num_outputs=1, num_in_hidden_layer_1=20), 1)
sys.stdout = open('results/NN-1-hidden-layers-Cancer-results-class-0.txt', 'w')
run_classification_experiment("data/breast-cancer-wisconsin.data.new.txt", NeuralNetwork(num_inputs=90, num_outputs=1, num_in_hidden_layer_1=20), 0)

sys.stdout = console_out
print("Soybean Data Set")

sys.stdout = open('results/NN-1-hidden-layers-soybean-small-results-class-D1.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", NeuralNetwork(num_inputs=204, num_outputs=1, num_in_hidden_layer_1=5), "D1")
sys.stdout = open('results/NN-1-hidden-layers-soybean-small-results-class-D2.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", NeuralNetwork(num_inputs=204, num_outputs=1, num_in_hidden_layer_1=5), "D2")
sys.stdout = open('results/NN-1-hidden-layers-soybean-small-results-class-D3.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", NeuralNetwork(num_inputs=204, num_outputs=1, num_in_hidden_layer_1=5), "D3")
sys.stdout = open('results/NN-1-hidden-layers-soybean-small-results-class-D4.txt', 'w')
run_classification_experiment("data/soybean-small.data.new.txt", NeuralNetwork(num_inputs=204, num_outputs=1, num_in_hidden_layer_1=5), "D4")

sys.stdout = console_out
print("Votes Data Set")

sys.stdout = open('results/NN-1-hidden-layers-house-votes-84-results-class-democrat.txt', 'w')
run_classification_experiment("data/house-votes-84.data.new.txt", NeuralNetwork(num_inputs=16, num_outputs=1, num_in_hidden_layer_1=5), "democrat")
sys.stdout = open('results/NN-1-hidden-layers-house-votes-84-results-class-republican.txt', 'w')
run_classification_experiment("data/house-votes-84.data.new.txt", NeuralNetwork(num_inputs=16, num_outputs=1, num_in_hidden_layer_1=5), "republican")

sys.stdout = console_out
print("Iris Data Set")

sys.stdout = open('results/NN-1-hidden-layers-iris-results-class-setosa.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", NeuralNetwork(num_inputs=24, num_outputs=1, num_in_hidden_layer_1=5), "Iris-setosa")
sys.stdout = open('results/NN-1-hidden-layers-iris-results-class-versicolor.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", NeuralNetwork(num_inputs=24, num_outputs=1, num_in_hidden_layer_1=5), "Iris-versicolor")
sys.stdout = open('results/NN-1-hidden-layers-iris-results-class-virginica.txt', 'w')
run_classification_experiment("data/iris.data.new.txt", NeuralNetwork(num_inputs=24, num_outputs=1, num_in_hidden_layer_1=5), "Iris-virginica")

sys.stdout = console_out
print("Glass Data Set")

sys.stdout = open('results/NN-1-hidden-layers-glass-results-class-1.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1, num_in_hidden_layer_1=5), 1)
sys.stdout = open('results/NN-1-hidden-layers-glass-results-class-2.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1, num_in_hidden_layer_1=5), 2)
sys.stdout = open('results/NN-1-hidden-layers-glass-results-class-3.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1, num_in_hidden_layer_1=5), 3)
sys.stdout = open('results/NN-1-hidden-layers-glass-results-class-5.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1, num_in_hidden_layer_1=5), 5)
sys.stdout = open('results/NN-1-hidden-layers-glass-results-class-6.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1, num_in_hidden_layer_1=5), 6)
sys.stdout = open('results/NN-1-hidden-layers-glass-results-class-7.txt', 'w')
run_classification_experiment("data/glass.data.new.txt", NeuralNetwork(num_inputs=54, num_outputs=1, num_in_hidden_layer_1=5), 7)

