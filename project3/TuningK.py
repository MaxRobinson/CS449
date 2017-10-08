""" Created by Max 10/8/2017 """

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import Experiments
from knn import Knn


# <editor-fold desc="Validation Curves">
def generate_validation_curves(x_axis_values, values_line_1, label_1, x_axis_label="", y_axis_label="MSE", title=""):
    plt.plot(x_axis_values, values_line_1, '-', label=label_1)
    # plt.plot(x_axis_values, values_line_2, '-', label=label_2)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    plt.legend(loc='best')

    plt.show()
    plt.close()

# </editor-fold>

# <editor-fold desc="Procedure">
def get_best_k_for_data_mse(data_set_path, number_of_runs):
    k_list = range(1, 11)

    train_mse_list_master = [0] *len(k_list)

    for x in range(30):
        train_mse_list = []
        for k in k_list:
            knn = Knn(k, False)
            average_mse = Experiments.run_experiment_regression(data_set_path, k, knn)
            train_mse_list.append(average_mse)

        for index in range(len(train_mse_list_master)):
            train_mse_list_master[index] += train_mse_list[index]
    for index in range(len(train_mse_list_master)):
        train_mse_list_master[index] /= 30
    generate_validation_curves(k_list, train_mse_list_master, "Average Mean Squared Error",
                               title="Number of K's vs AMSE", x_axis_label="# of k's", y_axis_label="MSE")


def get_best_k_for_data_error_rate_normal(data_set_path, number_of_runs):
    k_list = range(1, 11)

    train_mse_list_master = [0] * len(k_list)

    for x in range(number_of_runs):
        train_mse_list = []
        for k in k_list:
            knn = Knn(k, True)
            average_error_rate = Experiments.run_classification_experiment(data_set_path, k, knn)
            train_mse_list.append(average_error_rate)

        for index in range(len(train_mse_list_master)):
            train_mse_list_master[index] += train_mse_list[index]
    for index in range(len(train_mse_list_master)):
        train_mse_list_master[index] /= 30
    generate_validation_curves(k_list, train_mse_list_master, "Average Error Rate",
                               title="Number of K's vs Average Error Rate", x_axis_label="# of k's",
                               y_axis_label="Error Rate")


# </editor-fold>


get_best_k_for_data_mse("data/machine.data.new.txt", 20)

get_best_k_for_data_error_rate_normal("data/ecoli.data.new.txt", 20)
