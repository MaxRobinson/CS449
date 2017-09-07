from __future__ import division

import csv
import math
import random
import sys


# <editor-fold desc="Init Data">
def read_file(path=None):
    if path is None:
        path = 'data/breast-cancer-wisconsin.data.txt'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    for inner_list in csv_list:
        inner_list[:] = list(int(feature) for feature in inner_list)

    return csv_list
# </editor-fold>


# <editor-fold desc="Helpers">
def get_class(record):
    return record[-1]
# </editor-fold>


# <editor-fold desc="Winnow-2">
def init_weights(record_length):
    return [1]*record_length


def calculate_model_prediction(record, weights, threshold):
    running_sum = 0
    for feature, weight in zip(record, weights):
        running_sum += feature * weight
    if running_sum >= threshold:
        return 1
    else:
        return 0


def update_model(record, weights, threshold, alpha):
    prediction = calculate_model_prediction(record, weights, threshold)


def update_weights(record, weights, alpha, promote=True):
    new_weights = []
    for feature, weight in zip(record, weights):
        if feature == 1:
            if promote:
                new_weights.append(alpha * weight)
            else:
                new_weights.append(weight / alpha)
        else:
            new_weights.append(weight)

    return new_weights


def winnow_2(records, theta, learning_rate, positive_class=1, negative_class=0):
    weights = init_weights(len(records[0])-1)
    threshold = theta
    alpha = learning_rate
    # random.shuffle(records)

    one_third_data_length = int(math.floor(len(records)/3))
    training = records[:2*one_third_data_length]
    test = records[2*one_third_data_length:]

    count = 0
    print("Iteration #{0} (Initial Weights) : {1}".format(count, weights))
    for record in records:
        prediction = calculate_model_prediction(record, weights, threshold)
        actual_class = get_class(record)
        if prediction != actual_class:
            if prediction < actual_class:
                weights = update_weights(record, weights, alpha, True)
            else:
                weights = update_weights(record, weights, alpha, False)
        count += 1
        print("Iteration #{0} : {1}".format(count, weights))

    return weights
# </editor-fold>

# <editor-fold desc="Tests">
toy_values = read_file("data/toyExample.txt")
print(toy_values)

model = winnow_2(toy_values, .75, 2)
print(model)
# </editor-fold>
