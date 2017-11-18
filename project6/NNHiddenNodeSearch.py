""" Created by Max 11/17/2017 """

import numpy as np
import random

from NeuralNetworkFast import NeuralNetwork
from customCsvReader import CustomCSVReader


class hidden_layer1_node_search():
    def __init__(self, number_inputs, number_outputs):
        self.number_inputs = number_inputs
        self.number_outputs = number_outputs

    def find_optimal_hidden_node_1(self, start_number, max_number, step, dataset, positive_class_name):
        random.shuffle(dataset)
        processing_data = NeuralNetwork(self.number_inputs, self.number_outputs, num_in_hidden_layer_1=2)

        dataset = processing_data.pre_process(dataset, positive_class_name)

        training_set = dataset[:int(2*len(dataset)/3)]
        test_set = dataset[int(2*len(dataset)/3):]

        node_numbers = list(range(start_number, max_number + step, step))
        error_list = []

        for node_number in node_numbers:
            print("Trying node number: {}".format(node_number))
            nn = NeuralNetwork(self.number_inputs, self.number_outputs, num_in_hidden_layer_1=node_number)
            model = nn.learn(training_set)

            error_rate = self.calculate_error_rate(nn, model, test_set)

            error_list.append(error_rate)

        error_list = np.array(error_list)
        min_error_index = error_list.argmin()
        optimal_node_number = node_numbers[min_error_index]

        print(error_list)
        return optimal_node_number

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

            actual_prediction = prediction

            actuals.append(test_item[-1])
            if actual_prediction != test_item[-1]:
                num_errors += 1

        error_rate = num_errors / len(predictions)

        return error_rate


# <editor-fold desc="Cancer Data">
print("Cancer")
node_search = hidden_layer1_node_search(90, 1)

all_data = CustomCSVReader.read_file("data/breast-cancer-wisconsin.data.new.txt", float)
optimal_node_number = node_search.find_optimal_hidden_node_1(start_number=5, max_number=50,
                                                             step=5, dataset=all_data, positive_class_name=1)
print(optimal_node_number)
# </editor-fold>

# <editor-fold desc="Soybean">
print("Soybean")
node_search = hidden_layer1_node_search(204, 1)

all_data = CustomCSVReader.read_file("data/soybean-small.data.new.txt", float)
optimal_node_number = node_search.find_optimal_hidden_node_1(start_number=5, max_number=50,
                                                             step=5, dataset=all_data, positive_class_name="D1")
print(optimal_node_number)
# </editor-fold>

# <editor-fold desc="Votes">
print("Votes")
node_search = hidden_layer1_node_search(90, 1)

all_data = CustomCSVReader.read_file("data/house-votes-84.data.new.txt", float)
optimal_node_number = node_search.find_optimal_hidden_node_1(start_number=5, max_number=50,
                                                             step=5, dataset=all_data, positive_class_name="democrat")
print(optimal_node_number)
# </editor-fold>

# <editor-fold desc="iris">
print("Iris")
node_search = hidden_layer1_node_search(90, 1)

all_data = CustomCSVReader.read_file("data/iris.data.new.txt", float)
optimal_node_number = node_search.find_optimal_hidden_node_1(start_number=5, max_number=50,
                                                             step=5, dataset=all_data, positive_class_name="Iris-setosa")
print(optimal_node_number)
# </editor-fold>

# <editor-fold desc="Glass">
print("Glass")
node_search = hidden_layer1_node_search(90, 1)

all_data = CustomCSVReader.read_file("data/glass.data.new.txt", float)
optimal_node_number = node_search.find_optimal_hidden_node_1(start_number=5, max_number=50,
                                                             step=5, dataset=all_data, positive_class_name=3)
print(optimal_node_number)
# </editor-fold>




