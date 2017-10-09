""" Created by Max 10/6/2017 """
import random
import copy
from knn import Knn
from customCsvReader import CustomCSVReader

class CondensedNN:
    def __init__(self):
        """
        Use k-NN with k=1 to find the closes point in Z for condensing
        """
        self.knn = Knn(1, True)

    def condense(self, training_set):
        """
        Create the condensed model from the training set.

        Start with an empty set Z.
        While Z continues to change, repeat the following:
            For each data point in the training set x', find a point x^''in Z that is the closest point in Z to x'.
            If the labels of x^' and x'' are not the same, add x' to Z


        :param training_set: data points to select the condensed model from.
        :return: list of data points, the condensed model
        """
        random.shuffle(training_set)

        z = []
        previous_z = None

        while z != previous_z:
            previous_z = copy.deepcopy(z)
            for datapoint in training_set:
                if not z:
                    z.append(datapoint)

                nearest_z_class = self.find_nearest_z_class(z, datapoint)
                if datapoint[-1] != nearest_z_class:
                    z.append(datapoint)
        return z


    def find_nearest_z_class(self, z, datapoint):
        """
        Use the k-NN code with k = 1 to find the closest point in Z to the data point in question.
        :param z: Selected data points for condensed model
        :param datapoint: data point to query in Z
        :return: returns the label of the closest point
        """
        return self.knn.learn(z, datapoint)

