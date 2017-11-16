from __future__ import division
import numpy as np
import math


class NeuralNetwork:
    def __init__(self, num_inputs, num_in_hidden_layer_1, num_in_hidden_layer_2, num_outputs):
        self.layer1_weights = np.random.rand(num_in_hidden_layer_1, num_inputs)
        self.layer2_weights = np.random.rand(num_in_hidden_layer_2, num_in_hidden_layer_1)
        self.output_weight = np.random.rand(num_outputs, num_in_hidden_layer_2)

        self.weights = np.array([self.layer1_weights, self.layer2_weights, self.output_weight])

    def estimate_output(self, input_vector):
        self.results = []

        input_value = np.copy(input_vector)
        for layer in self.weights:
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


# <editor-fold desc="Tests">
nn = NeuralNetwork(90, 100, 30, 4)

input_vector = np.random.rand(90)
# input2 = np.ones((2, 2))
# layer = np.ones((1, 90))

# result = nn.compute_node_value(input_vector, layer)
result = nn.estimate_output(input_vector)

print(result)

# </editor-fold>

