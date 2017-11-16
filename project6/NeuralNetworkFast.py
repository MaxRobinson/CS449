from __future__ import division
import numpy as np
import math
import sys

from typing import Dict, Tuple, List


class NeuralNetwork:
    def __init__(self, num_inputs,  num_outputs, num_in_hidden_layer_1=None, num_in_hidden_layer_2=None):

        if num_in_hidden_layer_1 is None:
            self.output_weight = np.random.rand(num_outputs, num_inputs + 1)
            self.weights = np.array([self.output_weight])

        elif num_in_hidden_layer_1 is not None and num_in_hidden_layer_2 is None:
            self.layer1_weights = np.random.rand(num_in_hidden_layer_1, num_inputs + 1)
            self.output_weight = np.random.rand(num_outputs, num_in_hidden_layer_1 + 1)
            self.weights = np.array([self.layer1_weights, self.output_weight])

        else:
            self.layer1_weights = np.random.rand(num_in_hidden_layer_1, num_inputs + 1)
            self.layer2_weights = np.random.rand(num_in_hidden_layer_2, num_in_hidden_layer_1 + 1)
            self.output_weight = np.random.rand(num_outputs, num_in_hidden_layer_2 + 1)
            self.weights = np.array([self.layer1_weights, self.layer2_weights, self.output_weight])



    def estimate_output(self, input_vector):
        self.results = []

        input_value = np.copy(input_vector)
        for layer in self.weights:
            # Add Bias to the last position of the inputs for all layers
            input_value = np.append(input_value, [1])
            node_outputs = self.compute_node_value(input_value, layer)
            self.results.append(np.copy(node_outputs))
            input_value = node_outputs

        return self.results[-1]

    def compute_node_value(self, input_vector, layer):
        result_vector = np.inner(input_vector, layer)
        node_results = np.apply_along_axis(self.logistic_function, 0, result_vector)
        return node_results

    def logistic_function(self, value):
        return 1/(1 + math.e**(-1*value))



    def learn_model(self, data):
        """
        Perform Gradient Descent to learn the weights for the nodes.
        :param data: list of data points aka feature vectors
        :return: a matrix of weights for the edges between nodes
        """

        epsilon = .0001

        previous_error = sys.maxsize

        current_error = self.get_total_error_average(data)

        ittor_count = 0

        while not self.can_stop(current_error, previous_error, epsilon):
            for data_point in data:
                node_outputs = self.estimate_output(data_point)
                # do back prop to update weights

            previous_error = current_error
            current_error = self.get_total_error_average(data)

        if ittor_count % 1000 == 0:
            print("Count: {0} \n Current Error: {1}".format(ittor_count, current_error))

        ittor_count += 1

        return self.weights

    def can_stop(self, current_error, previous_error, epsilon):
        if abs(current_error - previous_error) > epsilon:
            return False
        return True

    def get_total_error_average(self, data: List[list]):
        """
        data[0][-1] are the expected values for the output of the network.
        For a network with more than 1 output node, this value will be an array.
        For a network with 1 output node, this value will be an value, corresponding the the actual value
        :param data:
        :return:
        """
        if type(data[0][-1]) != list:
            output_errors = np.zeros((len(data), 1))
        else:
            output_errors = np.zeros((len(data), len(data[0][-1])))

        for index in range(len(data)):
            node_outputs = self.estimate_output(data[index])
            error_per_output_node = self.calculate_output_node_error(data[index][-1], node_outputs)
            output_errors[index] = np.array(error_per_output_node)

        error_totals = np.sum(output_errors, axis=0)
        error_average = np.apply_along_axis(lambda error_total: error_total/len(data), 0, error_totals)
        mean_summed_error = np.sum(error_average)

        return mean_summed_error




    def calculate_output_node_error(self, actual_output_vector: list, predicted_output_vector: np.ndarray) -> list:
        if type(actual_output_vector) != list:
            actual_output_vector = [actual_output_vector]

        error_list = []
        for actual_value, predicted_value in zip(actual_output_vector, predicted_output_vector):

            error = predicted_value * (1 - predicted_value) * (actual_value - predicted_value)
            error_list.append(error)

        return error_list




# <editor-fold desc="Tests">
nn = NeuralNetwork(num_inputs=5, num_outputs=1, num_in_hidden_layer_1=3)

input_vector = np.random.rand(5)
# input2 = np.ones((2, 2))
# layer = np.ones((1, 90))

# result = nn.compute_node_value(input_vector, layer)
# result = nn.estimate_output(input_vector)
# print(result)

model = nn.learn_model(np.random.rand(1,5))

print(model)



# </editor-fold>

