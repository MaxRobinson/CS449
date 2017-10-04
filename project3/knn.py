from __future__ import division
import matplotlib.pyplot as plt
import csv, math, random, sys
import datetime
import numpy as np
import matplotlib.mlab as mlab

__author__ = 'Max'


# <editor-fold desc="Init Data">
def read_file(path=None):
    if path is None:
        path = 'concrete_compressive_strength.csv'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    records = list(map(lambda inner_list: map(lambda value: float(value), inner_list), csv_list))

    return records

# </editor-fold>

# <editor-fold desc="k-NN">


def feature_distance(datapoint_feature, query_feature):
    feature_diff = datapoint_feature - query_feature
    feature_diff *= feature_diff
    return feature_diff


def distance(datapoint, query):
    dist_sum = 0
    for datapoint_feature, query_feature in zip(datapoint, query):
        dist_sum += feature_distance(datapoint_feature, query_feature)

    dist = math.sqrt(dist_sum)

    return dist


def processing(k_nearest):
    y_sum = 0
    for dist, features in k_nearest:
        y_sum += features[-1]

    average = y_sum / len(k_nearest)

    return average


def add_to_smallest_distances(k_smallest_distances, distance_of_features, datapoint):
    if distance_of_features <= k_smallest_distances[-1][0]:
        k_smallest_distances[-1] = (distance_of_features, datapoint)
        k_smallest_distances.sort(key=lambda tup: tup[0])

    return k_smallest_distances


def knn(k, dataset, query):
    # distances = []
    k_smallest_distances = [(sys.maxint, []) for x in xrange(k)]

    for datapoint in dataset:
        datapoint_features_only = datapoint[:-1]
        distance_of_features = distance(datapoint_features_only, query)
        # distances.append((distance_of_features, datapoint))
        k_smallest_distances = add_to_smallest_distances(k_smallest_distances, distance_of_features, datapoint)

    # distances.sort(key=lambda tup: tup[0])
    # k_nearest = distances[:k]

    return processing(k_smallest_distances)
    # return processing(k_nearest)


def knn_with_query_set(k, dataset, query_set):
    results = []
    for item in query_set:
        results.append(knn(k, dataset, item))
    return results

# </editor-fold>

# <editor-fold desc="Validation Curves">
def generate_validation_curves(x_axis_values, values_line_1, values_line_2, label_1, label_2, x_axis_label="", y_axis_label="MSE", title=""):
    plt.plot(x_axis_values, values_line_1, '-', label=label_1)
    plt.plot(x_axis_values, values_line_2, '-', label=label_2)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    plt.legend(loc='best')

    plt.show()
    plt.close()

# </editor-fold>

# <editor-fold desc="Procedure">
def get_squared_error(predicted_value, item):
    actual_value = item[-1]
    squared_error = (predicted_value - actual_value)**2
    return squared_error


def calculate_mse(k, training_data, test_data):
    squared_error = []
    # for item in test_data:
    #     value = knn(k, training_data, item)
    #     squared_error.append(get_squared_error(value, item))
    predictions = knn_with_query_set(k, training_data, test_data)
    for prediction, test_item in zip(predictions, test_data):
        squared_error.append(get_squared_error(prediction, test_item))

    squared_error_sum = sum(squared_error)

    mse = squared_error_sum / len(squared_error)

    return mse


def get_best_k_for_data():
    k_list = range(1, 11)

    dataset = read_file()
    random.shuffle(dataset)

    two_thirds = 2 * int(math.floor(len(dataset)/3))
    training_data = dataset[:two_thirds]
    test_data = dataset[two_thirds:]

    test_mse_list = []
    train_mse_list = []

    for k in k_list:
        test_mse_list.append(calculate_mse(k, training_data, test_data))
        train_mse_list.append(calculate_mse(k, training_data, training_data))

    generate_validation_curves(k_list, train_mse_list, test_mse_list, "Training Data Mean Squared Error",
                               "Test Data Mean Squared Error", title="Number of K's vs MSE", x_axis_label="# of k's")

    min_error = min(test_mse_list)
    return test_mse_list.index(min_error) + 1


def learning_curves(k):
    dataset = read_file()
    random.shuffle(dataset)

    two_thirds = 2 * int(math.floor(len(dataset)/3))
    training_data = dataset[:two_thirds]
    test_data = dataset[two_thirds:]

    percent_increase = range(5, 105, 5)

    training_mse_list = []
    test_mse_list = []

    for percentage_of_data in percent_increase:
        amount_of_data = int(math.floor(len(training_data) * (percentage_of_data * .01)))
        subset_data = training_data[:amount_of_data]

        test_mse_list.append(calculate_mse(k, subset_data, test_data))
        training_mse_list.append(calculate_mse(k, subset_data, subset_data))

    generate_validation_curves(percent_increase, training_mse_list, test_mse_list, "Training Data Mean Squared Error",
                               "Test Data Mean Squared Error", title="Learning Curve for k={}".format(k),
                               x_axis_label="Percentage of data Used")

# </editor-fold>


# <editor-fold desc="Cross Validation">

def calc_standard_deviation(average, list_of_values):
    sd = 0
    for x in list_of_values:
        sd += (x - average)**2

    sd /= len(list_of_values)
    sd = math.sqrt(sd)

    return sd


def plot_normal_distribution(mu, sigma):
    variance = sigma**2

    x = np.linspace(mu-3*variance, mu+3*variance, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma))

    plt.show()
    plt.close()


def cross_validation(k):
    dataset = read_file()
    random.shuffle(dataset)

    tenth_length = int(math.floor(len(dataset)/10))

    cross_validation_dataset = []
    for i in range(0, len(dataset), tenth_length):
        cross_validation_dataset.append(dataset[i:i+tenth_length])

    # run cross validation
    mse_list = []
    for i in xrange(10):
        test_set = cross_validation_dataset[i]
        training_set = cross_validation_dataset[:i] + cross_validation_dataset[i+1:]
        training_set = [item for sublist in training_set for item in sublist]

        mse_list.append(calculate_mse(k, training_set, test_set))

    average_mse = sum(mse_list) / len(mse_list)
    sd = calc_standard_deviation(average_mse, mse_list)

    print "Average MSE: {}".format(average_mse)
    print "Standard Deviation: {}".format(sd)
    # print mse_list
    # plot_normal_distribution(average_mse, sd)

# </editor-fold>


# <editor-fold desc="Test">
# print read_file()

# x = [540.0 ,0.0 ,0.0 ,162.0 ,2.5 ,1040.0 ,676.0 ,28 ,79.99 ]
# y = [540.0 ,0.0 ,0.0 ,162.0 ,2.5 ,1055.0 ,676.0 ,28 ,61.89 ]
# #
# # print knn(1, [x], y)
#
# x_dist = (15, x)
# y_dist = (5, y)
#
# print processing([x_dist, y_dist])
#
#
# x1 = [.73, .23, .89, .18]
# y1 = [.34, .32, .65, .24]
#
# print distance(x1, y1)

best_k = get_best_k_for_data()
print best_k

learning_curves(best_k)


cross_validation(2)

# </editor-fold>
