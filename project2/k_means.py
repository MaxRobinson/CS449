""" Created by Max 9/22/2017 """
import random
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from customCsvReader import CustomCSVReader
from FischerScore import fisher_score


class KMeans:

    def __init__(self, k):
        """
        Init with K clusters
        :param k: int, k clusters.
        """
        self.k = k


    def learn(self, selected_features, data_training, mu=.00001):
        """
        The main work horse for the Kmeans algorithm

        First thing we do, is given the training data, we strip out all features that we don't need.
        We then run the kMeans algorithm on the data.

        :param selected_features: a list of unique integers i.e [0,1,2] that specify the features to use.
        :param data_training: list of list trainig data
        :param mu: the difference in means before stopping.
        :return: returns a list of mean vectors.
        """
        data = self.data_munge(selected_features, data_training)

        means = self.init_k_means(data, self.k)
        previous_means = None

        # itter = 0
        while not self.has_stopped_changing(means, previous_means, mu):
            # print(itter)
            clusters = self.create_clusters(self.k)
            previous_means = copy.deepcopy(means)

            for datapoint in data:
                closest_mean_index = self.argmin_cluster(datapoint, means)
                clusters[closest_mean_index].append(datapoint)

            means = self.calculate_means(clusters, means)
            # itter += 1

        return means


    def evaluate(self, means, selected_features, data_test):
        """
        Evaluates a given set of mean vectors using the Fisher Score
        Given the means, selected_features and data:
        Trim the data to just the selected features, then caclulate the clusters based on the means.
        Calculate the socre by passing the clusters and the means to the Fisher score.

        :param means: list of mean vectors of length "selected Features"
        :param selected_features: the features selected to score.
        :param data_test: list of list, data points.
        :return: the Fisher Score.
        """
        data = self.data_munge(selected_features, data_test)
        clusters = self.create_clusters(len(means))

        for data in data:
            cluster_index = self.argmin_cluster(data, means)
            clusters[cluster_index].append(data)


        score = fisher_score(means, clusters)
        # print("Fisher Score = {}".format(score))
        return score

    def get_clusters_for_means(self, means, selected_features, data_to_cluster):
        """
        A helper function that, given mean vectors, will cluster all of the data into clusters based on those means.
        Returns the clustered data
        :param means:
        :param selected_features:
        :param data_to_cluster:
        :return: dict of clusterd data in k clusters
        """
        data = self.data_munge(selected_features, data_to_cluster)
        clusters = self.create_clusters(len(means))

        for data_index in range(len(data)):
            datum = data[data_index]
            cluster_index = self.argmin_cluster(datum, means)
            clusters[cluster_index].append(data_to_cluster[data_index])

        return clusters


    def calculate_means(self, clusters, means):
        """
        Helper function to calculate all the mean vector for each cluster
        :param clusters: dict of clusters
        :param means: list of means
        :return: updated list of means
        """
        for cluster_number, datapoints_in_cluster in clusters.items():
            new_mean = self.calculate_mean_vector(datapoints_in_cluster, len(means[0]))
            means[cluster_number] = new_mean
        return means

    def calculate_mean_vector(self, datapoints_in_cluster, vector_length):
        """
        Calculates the mean vector for a cluster.

        :param datapoints_in_cluster: list of list
        :param vector_length: int
        :return: the mean vector for the cluster
        """
        mean = [0]*vector_length
        for datapoint in datapoints_in_cluster:
            for feature_index in range(len(datapoint)):
                mean[feature_index] += datapoint[feature_index]

        for mean_feature_index in range(len(mean)):
            # if not datapoints_in_cluster:
            #     continue
            mean[mean_feature_index] = mean[mean_feature_index] / float(len(datapoints_in_cluster))

        return mean

    def argmin_cluster(self, datapoint, means):
        """
        Gets the cluster that is the closests to a data point.

        the corresponding mean vector corresponds to the cluster the point belongs too.

        :param datapoint: list of float
        :param means: a list of mean vectors.
        :return: int index of Selected_mean
        """
        min_distance = sys.maxsize
        selected_mean_index = -1

        for index_of_mean in range(len(means)):
            mean = means[index_of_mean]
            dist = self.distance(datapoint, mean)
            if dist < min_distance:
                min_distance = dist
                selected_mean_index = index_of_mean

        return selected_mean_index

    def distance(self, datapoint, mean):
        """
        Euclidean distance measure, for all dimensions in data point, from point to mean vector
        :param datapoint: list float
        :param mean: list float
        :return: float, distance
        """
        running_sum = 0
        for feature_value, feature_value_mean in zip(datapoint, mean):
            running_sum += (feature_value - feature_value_mean)**2

        return math.sqrt(running_sum)

    # <editor-fold desc="Helpers">
    def has_stopped_changing(self, means, previous_means, mu):
        """
        Used to tell if all means have stopped changing.
        The difference in all values in the mean have to be less than mu for that mean.
        :param means: list of mean vectors
        :param previous_means: the means list from the previous itteration.
        :param mu: value to stop changing at. float
        :return: boolean, true if has stopped changing.
        """
        if previous_means is None:
            return False
        for mean, previous_mean in zip(means, previous_means):
            for mean_feature_value, previous_mean_feature_value in zip(mean, previous_mean):
                if mean_feature_value - previous_mean_feature_value > mu:
                    return False
        return True


    def create_clusters(self, number_of_clusters):
        """
        creates a dictionary of clusters, where a single cluster is a list
        :param number_of_clusters: int
        :return: dict cluster
        """
        cluster = {}
        for cluster_number in range(number_of_clusters):
            cluster[cluster_number] = []
        return cluster

    def init_k_means(self, data, num_k_s):
        """
        Init's k means. Choosed k means based on randomly selecting points in the data set.
        :param data: list list
        :param num_k_s: int
        :return: list of k means ( selected data poinst to start the mean at).
        """
        means = []
        data_length = len(data)

        for x in range(num_k_s):
            random_data_index = random.randint(0, data_length-1)
            selected_datapoint = data[random_data_index]

            while selected_datapoint in means:
                random_data_index = random.randint(0, data_length-1)
                selected_datapoint = data[random_data_index]
            means.append(selected_datapoint)

        return means

    # </editor-fold>


    def data_munge(self, selected_features, data):
        """
        Modifies the data set to only contain the selected features.

        :param selected_features: list of ints of selected features.
        :param data: list of list of points
        :return: munged data, list of list.
        """
        new_data = []
        for data_point in data:
            new_data_point = []
            for selected_feature in selected_features:
                new_data_point.append(data_point[selected_feature])
            new_data.append(new_data_point)
        return new_data

    """
    KMeans(D, k):
        means = init_k_means()
        clusters = [[]];
        do: 
            for datapoint in data:
                c = argmin_u_j distance(datapoint, means_j)
                assign x_i to the cluster c
            
            recalculate all U_j based on new clusters
        until no change in Means
        
        return means 
    """

