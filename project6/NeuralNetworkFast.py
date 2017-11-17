from __future__ import division
import numpy as np
import math
import sys
import copy

from typing import Dict, Tuple, List


class NeuralNetwork:
    def __init__(self, num_inputs,  num_outputs, num_in_hidden_layer_1=None, num_in_hidden_layer_2=None):
        self.layer_outputs = []

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
        self.layer_outputs = []

        input_value = np.copy(input_vector[:-1])
        for layer in self.weights:
            # Add Bias to the last position of the inputs for all layers
            input_value = np.append(input_value, [1])
            node_outputs = self.compute_node_value(input_value, layer)
            self.layer_outputs.append(np.copy(node_outputs))
            input_value = node_outputs

        return copy.copy(self.layer_outputs[-1])

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

        epsilon = .000001
        previous_error = sys.maxsize
        current_error = self.get_total_error_average(data)

        ittor_count = 0
        while not self.can_stop(current_error, previous_error, epsilon):
            for data_point in data:
                node_outputs = self.estimate_output(data_point)
                # do back prop to update weights
                self.weights = self.back_prop(self.weights, node_outputs, np.array(self.layer_outputs), data_point[-1], data_point[:-1])

            previous_error = current_error
            current_error = self.get_total_error_average(data)

            # if ittor_count % 1000 == 0:
            #     print("Count: {0} \n Current Error: {1}".format(ittor_count, current_error))
            print("Count: {0} \n Current Error: {1}".format(ittor_count, current_error))
            ittor_count += 1

        return self.weights

    def can_stop(self, current_error, previous_error, epsilon):
        if abs(current_error - previous_error) > epsilon:
            return False
        return True

    def back_prop(self, weights: np.ndarray, node_outputs: np.ndarray, layer_outputs:np.ndarray,
                  actual_result, data_input):

        weight_error_matrix = self.calculate_layer_errors(weights, node_outputs, layer_outputs, actual_result)
        new_weights = self.update_wights(np.copy(weights), data_input, layer_outputs, weight_error_matrix)
        return new_weights

    def calculate_layer_errors(self, weights: np.ndarray, node_outputs: np.ndarray, layer_outputs:np.ndarray,
                               actual_result):

        layer_errors = np.empty(len(weights), dtype=object)

        for layer_index in range(len(weights)-1, -1, -1):
            # if this is the output later, calculate the error differently
            if layer_index == len(weights)-1:
                # deltas
                output_layer_weights = self.calculate_output_node_error(actual_result, node_outputs)
                layer_errors[layer_index] = output_layer_weights
            else:
                # Get the error for a hidden layer.
                # need all weights, the layer_index, the layer_outputs and the delta/error for the previous layer.
                hidden_layer_error = self.calculate_hidden_node_error(weights[layer_index+1], layer_outputs[layer_index],
                                                 layer_errors[layer_index + 1])

                # Remove the hidden layer error for the Bias node (IT IS NOT REUSED)
                layer_errors[layer_index] = np.delete(hidden_layer_error, len(hidden_layer_error)-1)

        return layer_errors

    def calculate_hidden_node_error(self, previous_layer_weights: np.ndarray, layer_outputs: np.ndarray, layer_errors: np.ndarray):
        # calc contribution to error

        # contrib_to_error = layer_errors * weights
        # contrib_to_error = np.sum(contrib_to_error, axis=1)

        # contrib_to_error = np.inner(layer_errors, previous_layer_weights)
        contrib_to_error = np.empty(len(previous_layer_weights[0]), dtype=object)

        for node_index in range(len(previous_layer_weights[0])):
            # for node ith in the hidden layer, get the ith, weight for every node that it was applied to.
            weights_that_contribed = previous_layer_weights[:, node_index]
            theta_error = np.inner(layer_errors, weights_that_contribed)
            contrib_to_error[node_index] = theta_error

        # layer_output_mults = y_hat (1-y_hat)
        layer_output_with_bias = np.append(layer_outputs, [1]) # add BIAS back to layer outputs
        layer_output_mults = np.apply_along_axis(lambda y_hat: y_hat * (1-y_hat), 0, layer_output_with_bias)
        hidden_layer_errors = layer_output_mults * contrib_to_error

        return hidden_layer_errors

    def update_wights(self, weights: np.ndarray, data_input: np.ndarray, layer_outputs: np.ndarray, weight_error_matrix: np.ndarray, alpha=.1):
        for layer_index in range(len(weights)-1, -1, -1):
            layer_weights = weights[layer_index]

            # get the previous layer INPUTS
            if layer_index == 0:
                layer_output = np.array(data_input)
            else:
                # get the previous layer INPUTS
                layer_output = layer_outputs[layer_index - 1]

            layer_error = weight_error_matrix[layer_index]

            # Add BIAS to input values, in last position, as done in estimating
            layer_output = np.append(layer_output, [1])

            # weight_delta = layer_output * layer_error
            layer_output = layer_output.reshape(1, len(layer_output))
            layer_error = layer_error.reshape(1, len(layer_error))
            weight_delta = layer_output * layer_error.transpose()

            weight_delta = alpha * weight_delta

            weights[layer_index] = layer_weights + weight_delta

        return weights

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
            output_errors[index] = error_per_output_node

        error_totals = np.sum(output_errors, axis=0)
        error_average = np.apply_along_axis(lambda error_total: error_total/len(data), 0, error_totals)
        mean_summed_error = np.sum(error_average)

        return mean_summed_error

    def calculate_output_node_error(self, actual_output_vector: list, predicted_output_vector: np.ndarray) -> np.ndarray:
        if type(actual_output_vector) != list:
            actual_output_vector = [actual_output_vector]

        error_list = []
        for actual_value, predicted_value in zip(actual_output_vector, predicted_output_vector):

            error = predicted_value * (1 - predicted_value) * (actual_value - predicted_value)
            error_list.append(error)

        return np.array(error_list)




# <editor-fold desc="Tests">
# nn = NeuralNetwork(num_inputs=2, num_outputs=2, num_in_hidden_layer_1=3, num_in_hidden_layer_2=2)
nn = NeuralNetwork(num_inputs=2, num_outputs=2, num_in_hidden_layer_1=3)

# input_vector = np.random.rand(5)
# input2 = np.ones((2, 2))
# layer = np.ones((1, 90))

# result = nn.compute_node_value(input_vector, layer)
# result = nn.estimate_output(input_vector)
# print(result)

# model = nn.learn_model(np.random.rand(1,5))
#
# print(model)

# nn.layer1_weights = np.array([[.26, -.42, .01], [.78, .19, -.05], [-.23, .37, .42]])
# nn.output_weight = np.array([[.61, .12, -.9, .2], [.28, -.34, .10, .3]])
# nn.weights = np.array([nn.layer1_weights, nn.output_weight])
#
# estimate = nn.estimate_output([.52, -.97, [1,0]])
# layer_outputs = nn.layer_outputs
#
# print(estimate)
# print(layer_outputs)
#
# new_weights = nn.back_prop(nn.weights, estimate, np.array(layer_outputs), [1, 0], [.52, -.97])
# print(new_weights)

nn.learn_model([[.52, -.97, [1,0]], [.6, -1, [1,0]], [1, -.77, [1,0]], [.3, -.31, [1,0]]])

# </editor-fold>

