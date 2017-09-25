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
        self.k = k


    def learn(self, selected_features, data_training, mu=.00001):
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
        data = self.data_munge(selected_features, data_test)
        clusters = self.create_clusters(len(means))

        for data in data:
            cluster_index = self.argmin_cluster(data, means)
            clusters[cluster_index].append(data)


        score = fisher_score(means, clusters)
        # print("Fisher Score = {}".format(score))
        return score




    def calculate_means(self, clusters, means):
        for cluster_number, datapoints_in_cluster in clusters.items():
            new_mean = self.calculate_mean_vector(datapoints_in_cluster, len(means[0]))
            means[cluster_number] = new_mean
        return means

    def calculate_mean_vector(self, datapoints_in_cluster, vector_length):
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
        running_sum = 0
        for feature_value, feature_value_mean in zip(datapoint, mean):
            running_sum += (feature_value - feature_value_mean)**2

        return math.sqrt(running_sum)

    # <editor-fold desc="Helpers">
    def has_stopped_changing(self, means, previous_means, mu):
        if previous_means is None:
            return False
        for mean, previous_mean in zip(means, previous_means):
            for mean_feature_value, previous_mean_feature_value in zip(mean, previous_mean):
                if mean_feature_value - previous_mean_feature_value > mu:
                    return False
        return True


    def create_clusters(self, number_of_clusters):
        cluster = {}
        for cluster_number in range(number_of_clusters):
            cluster[cluster_number] = []
        return cluster

    def init_k_means(self, data, num_k_s):
        means = []
        data_length = len(data)

        for x in range(num_k_s):
            random_data_index = random.randint(0, data_length-1)
            selected_datapoint = data[random_data_index]

            while selected_datapoint in means:
                random_data_index = random.randint(0, data_length)
                selected_datapoint = data[random_data_index]
            means.append(selected_datapoint)

        return means

    # </editor-fold>


    def data_munge(self, selected_features, data):
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

def closest_cluster(datapoint, means):
    min_distance = sys.maxsize
    selected_mean_index = -1

    for index_of_mean in range(len(means)):
        mean = means[index_of_mean]
        dist = distance(datapoint, mean)
        if dist < min_distance:
            min_distance = dist
            selected_mean_index = index_of_mean

    return selected_mean_index

def distance(datapoint, mean):
    running_sum = 0
    for feature_value, feature_value_mean in zip(datapoint, mean):
        running_sum += (feature_value - feature_value_mean)**2

    return math.sqrt(running_sum)

#
# all_data = CustomCSVReader.read_file("data/iris.data.txt", float)
#
# kMeans = KMeans(3)
# means = kMeans.learn([2], all_data)
# print(means)
#
#
# trimmed_data = kMeans.data_munge([2], all_data)
#
# clusters = kMeans.create_clusters(3)
#
# for data in trimmed_data:
#     cluster_index = closest_cluster(data, means)
#     clusters[cluster_index].append(data)
#
#
# plt.plot(clusters[0], np.zeros_like(clusters[0]), 'x', color='red')
# plt.plot(clusters[1], np.zeros_like(clusters[1]), 'x', color='blue')
# plt.plot(clusters[2], np.zeros_like(clusters[2]), 'x', color='green')
# plt.show()


