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
        new_values = []
        for feature in inner_list:
            try:
                new_feature = int(feature)
                new_values.append(new_feature)
            except Exception:
                new_values.append(feature)

        inner_list[:] = new_values

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


def winnow_2(training_set, theta, learning_rate):
    weights = init_weights(len(training_set[0]) - 1)
    threshold = theta
    alpha = learning_rate
    random.shuffle(training_set)

    count = 0
    print("Iteration #{0} (Initial Weights) : {1}".format(count, weights))
    for record in training_set:
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


# <editor-fold desc="Verification">
def evaluate_winnow_2(model, test, threshold):
    results = []
    error = 0
    for record in test:
        prediction = calculate_model_prediction(record, model, threshold)
        results.append(prediction)
        if prediction != get_class(record):
            error += 1

    error_rate = error/len(test)
    return (error_rate, results, model)


def test_winnow_2(records):
    # records = read_file(data_set_name)
    random.shuffle(records)

    one_third_data_length = int(math.floor(len(records)/3))
    training = records[:2*one_third_data_length]
    test = records[2*one_third_data_length:]

    threshold = .75
    model = winnow_2(training, .75, 2)

    results = evaluate_winnow_2(model, test, threshold)
    return results

# </editor-fold>

# <editor-fold desc="preprocess">
def pre_process(data, positive_class_name):
    new_data = []
    for record in data:
        current_class = get_class(record)
        if current_class == positive_class_name:
            record[-1] = 1
        else:
            record[-1] = 0
        new_data.append(record)
    return new_data
# </editor-fold>

# <editor-fold desc="Experiment">
def run_experiment(data_set_path, positive_class_name):
    print("Running {0} Experiment with positive class {1}".format(data_set_path, positive_class_name))
    records = read_file(data_set_path)
    records = pre_process(records, positive_class_name)
    # results = test_winnow_2(test_records)

    random.shuffle(records)

    one_third_data_length = int(math.floor(len(records)/3))
    training = records[:2*one_third_data_length]
    test = records[2*one_third_data_length:]

    threshold = .75

    model = winnow_2(training, .75, 2)

    results = evaluate_winnow_2(model, test, threshold)
    print("Results: \n model: {0} \n classifications on test set: {1}".format(results[2], results[1]))
    print("Error Rate = {} \n".format(results[0]))



# </editor-fold>


# sys.stdout = open('TestOutput', 'w')
#
# run_experiment("data/breast-cancer-wisconsin.data.new.txt", 1)
# run_experiment("data/breast-cancer-wisconsin.data.new.txt", 0)
#
# run_experiment("data/soybean-small.data.new.txt", "D1")
# run_experiment("data/soybean-small.data.new.txt", "D2")
# run_experiment("data/soybean-small.data.new.txt", "D3")
# run_experiment("data/soybean-small.data.new.txt", "D4")
#
# run_experiment("data/house-votes-84.data.new.txt", "democrat")
# run_experiment("data/house-votes-84.data.new.txt", "republican")

run_experiment("data/iris.data.new.txt", "Iris-setosa")
run_experiment("data/iris.data.new.txt", "Iris-versicolor")
run_experiment("data/iris.data.new.txt", "Iris-virginica")


