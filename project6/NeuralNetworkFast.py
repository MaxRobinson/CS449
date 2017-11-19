from __future__ import division
import numpy as np
import math
import sys
import copy

from typing import Dict, Tuple, List


class NeuralNetwork:
    """
    A neural network implimentation that uses Numpy Matricies to do most of the calculations
    """

    def __init__(self, num_inputs,  num_outputs, num_in_hidden_layer_1=None, num_in_hidden_layer_2=None):
        """
        Assign the structure of the network.
        Give the number of inputs, outputs, and number per hidden layer.
        Then constructs the network.
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_in_hidden_layer_1 = num_in_hidden_layer_1
        self.num_in_hidden_layer_2 = num_in_hidden_layer_2

        self.init()

    def init(self):
        """
        allows for network to be reset in between runs.
        Randomly generates weights that conform the the specified number of weights in each layer.
        Actually creaets the wieghts for the network

        Self.weights stores the matrix of weights in order from input to output
        """

        num_inputs = self.num_inputs
        num_outputs = self.num_outputs
        num_in_hidden_layer_1 = self.num_in_hidden_layer_1
        num_in_hidden_layer_2 = self.num_in_hidden_layer_2


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
        """
        This is the forward pass of the network.
        Applies the network to the input vector to get the estimated output.
        Saves the layer outputs per layer so that backProp can be done after words if training.
        """
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
        """
        Computes the output value for a given layer given a vector input.
        A layer is a layer of WEIGHTS not nodes. Allows for the matrix mulitiplication to work.
        Applies the activation function to the nodes as well.
        Hands back a vector of node outputs.
        """
        result_vector = np.inner(input_vector, layer)
        node_results = np.apply_along_axis(self.logistic_function, 0, result_vector)
        return node_results

    def logistic_function(self, value):
        return 1/(1 + math.e**(-1*value))

    def learn(self, data):
        return self.learn_model(data)

    def learn_model(self, data):
        """
        Perform Gradient Descent to learn the weights for the nodes.
        Does online training where backprop is done after each datapoint passes through the network.

        Also ensures that the error is decreasing and not increasing.
        """

        epsilon = .000001
        previous_error = sys.maxsize
        current_error = self.get_total_error_average(data)

        ittor_count = 0
        while (not self.can_stop(current_error, previous_error, epsilon)) and abs(current_error) < abs(previous_error):
            for data_point in data:
                node_outputs = self.estimate_output(data_point)
                # do back prop to update weights
                self.weights = self.back_prop(self.weights, node_outputs, np.array(self.layer_outputs), data_point[-1], data_point[:-1])

            previous_error = current_error
            current_error = self.get_total_error_average(data)

            # if ittor_count % 1000 == 0:
            #     print("Count: {0} \n Current Error: {1}".format(ittor_count, current_error))
            # print("Count: {0} \n Current Error: {1}".format(ittor_count, current_error))
            ittor_count += 1
            #
            # if abs(current_error) > abs(previous_error):
            #     print("SOMETHING WENT WRONG")
            #     print("Current error: {}".format(abs(current_error)))
            #     print("previous error: {}".format(abs(previous_error)))

        # print("Count: {0} \n Current Error: {1}".format(ittor_count, current_error))
        return self.weights

    def can_stop(self, current_error, previous_error, epsilon):
        """
        Simple check to see if the network can stop
        """
        if abs(current_error - previous_error) > epsilon:
            return False
        return True

    def back_prop(self, weights: np.ndarray, node_outputs: np.ndarray, layer_outputs:np.ndarray,
                  actual_result, data_input):
        """
        Main work horse for the back propagation algorithm.
        First we calculate all of the errors for each layer, ( a matrix of those)
        Then apply the update weights rule based on the errors.
        """

        weight_error_matrix = self.calculate_layer_errors(weights, node_outputs, layer_outputs, actual_result)
        new_weights = self.update_wights(np.copy(weights), data_input, layer_outputs, weight_error_matrix)
        return new_weights

    def calculate_layer_errors(self, weights: np.ndarray, node_outputs: np.ndarray, layer_outputs:np.ndarray,
                               actual_result):
        """
        calculate the error terms for the different weight layers
        output layer uses a different error calculation than the rest.
        Errors are added to the matrix
        Errors for hidden layer bias nodes are removed since they do not propagate backwards
        """

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
        """
        Calculates the error for the hidden nodes
        Takes into accout the contribution to different output nodes.
        Then the derivative of the activation function is applied to all of the sum of error contributions.
        """
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
        """
        Update weights given the error matrix.
        Updates the weights based on the input input from the input to the layer from the previous layer. (closer to output)
        """
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
        """
        if type(data[0][-1]) != list:
            output_errors = np.zeros((len(data), 1))
        else:
            output_errors = np.zeros((len(data), len(data[0][-1])))

        for index in range(len(data)):
            node_outputs = self.estimate_output(data[index])
            # error_per_output_node = self.calculate_output_node_error(data[index][-1], node_outputs)
            squared_error = self.calc_output_node_squared_error(data[index][-1], node_outputs)
            output_errors[index] = squared_error

        error_totals = np.sum(output_errors, axis=0)
        error_average = np.apply_along_axis(lambda error_total: error_total/len(data), 0, error_totals)
        mean_summed_error = np.sum(error_average)

        return abs(mean_summed_error)

    def calc_output_node_squared_error(self, actual_output_vector: list, predicted_output_vector: np.ndarray):
        """
        Calculates the output node squared error.
        Error is cauclated for all output nodes.
        """
        if type(actual_output_vector) != list:
            actual_output_vector = [actual_output_vector]

        error_list = []

        for actual_value, predicted_value in zip(actual_output_vector, predicted_output_vector):
            error_list.append((actual_value - predicted_value)**2)

        error_sum = sum(error_list) / float(2)
        return error_sum



    def calculate_output_node_error(self, actual_output_vector: list, predicted_output_vector: np.ndarray) -> np.ndarray:
        """
        Caclulates the derivative of the error function and gives the error of the output node
        that is to be used during UPDATING / backpropagation.
        """
        if type(actual_output_vector) != list:
            actual_output_vector = [actual_output_vector]

        error_list = []
        for actual_value, predicted_value in zip(actual_output_vector, predicted_output_vector):

            error = predicted_value * (1 - predicted_value) * (actual_value - predicted_value)
            error_list.append(error)

        return np.array(error_list)

    # <editor-fold desc="Classify">
    def classify(self, nn_model, test_data):
        """
        classify based on one's hot encoding or a single value for 1 node output.
        Use a .5 cutoff range to deterine if the value is part of the class or not.
        Works for both arrays and single vlues.
        """
        self.weights = nn_model

        results = []
        for data_point in test_data:
            estimate = self.estimate_output(data_point)

            if len(estimate) == 1:
                if estimate > .5:
                    results.append(1)
                else:
                    results.append(0)
            else:
                max_index = np.array(estimate).argmax()
                result = np.zeros(len(estimate))
                result[max_index] = 1
                results.append(result.tolist())

        return results






    # </editor-fold>

    # <editor-fold desc="Preprocess Data">
    """
        Pre-processes the data for a given test run. 

        The data is preprocessed by taking a positive class label, and modifying the in memory data to replace the 
        positive_class_name with a 1, and all other classification names as 0, negative. 

        This allows for easier binary classificaiton 

        input: 
        + data: list of feature vecotrs
        + positive_class_name: Stirng, class to be the positive set.

        """

    def pre_process(self, data, positive_class_name):
        new_data = []
        for record in data:
            current_class = record[-1]
            if current_class == positive_class_name:
                record[-1] = 1
            else:
                record[-1] = 0
            new_data.append(record)
        return new_data

    # </editor-fold>




# <editor-fold desc="Tests">
# nn = NeuralNetwork(num_inputs=2, num_outputs=1, num_in_hidden_layer_1=5, num_in_hidden_layer_2=2)
# nn = NeuralNetwork(num_inputs=2, num_outputs=1, num_in_hidden_layer_1=3)
# nn = NeuralNetwork(num_inputs=2, num_outputs=1)
# nn = NeuralNetwork(num_inputs=2, num_outputs=1, num_in_hidden_layer_1=3)

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

# nn.learn_model([[.52, -.97, 1], [.6, -1, 1], [1, -.77, 1], [.3, -.31, 1]])

# </editor-fold>

