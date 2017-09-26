""" Created by Max 9/22/2017 """
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from customCsvReader import CustomCSVReader
from FischerScore import fisher_score

class HAC:

    def __init__(self, num_clusters):
        self.k = num_clusters


    def learn(self, selected_features, data_training):
        data = self.data_munge(selected_features, data_training)

        distance_matrix = np.zeros((len(data), len(data)))
        distance_matrix = self.compute_initial_distances(distance_matrix, data)

        clusters = self.init_clusters(len(data))

        while len(distance_matrix) > self.k:
            distance_matrix, clusters = self.agglomerate(distance_matrix, clusters)
            # print(clusters)

        # print("Initial distance Matrix: ")
        # print(distance_matrix)
        # print(clusters)
        clusters_with_datapoints = self.create_datapoint_clusters(clusters, data)

        return clusters_with_datapoints

    # <editor-fold desc="Evaluate">
    def evaluate(self, clustered_datapoints, selected_features, data_test):
        data = self.data_munge(selected_features, data_test)
        num_features = len(selected_features)

        model_clusters = self.create_clusters(len(clustered_datapoints))

        # Calculate means for fisher score
        for index_of_cluster_of_datapoints in range(len(clustered_datapoints)):
            cluster_of_datapoints = clustered_datapoints[index_of_cluster_of_datapoints]
            model_clusters[index_of_cluster_of_datapoints] = cluster_of_datapoints

        means = [[0] * num_features] * self.k
        means = self.calculate_means(model_clusters, means)

        # Cluster all of the datapoints based on means
        full_clusters = self.create_clusters(len(means))
        for data in data:
            cluster_index = self.argmin_cluster(data, means)
            full_clusters[cluster_index].append(data)

        score = fisher_score(means, full_clusters)
        # print("Fisher Score = {}".format(score))
        return score

    def calculate_means(self, clusters, means):
        for cluster_number, datapoints_in_cluster in clusters.items():
            new_mean = self.calculate_mean_vector(datapoints_in_cluster, len(means[0]))
            means[cluster_number] = new_mean
        return means

    def calculate_mean_vector(self, datapoints_in_cluster, vector_length):
        mean = [0] * vector_length
        for datapoint in datapoints_in_cluster:
            for feature_index in range(len(datapoint)):
                mean[feature_index] += datapoint[feature_index]

        for mean_feature_index in range(len(mean)):
            # if not datapoints_in_cluster:
            #     continue
            mean[mean_feature_index] = mean[mean_feature_index] / float(len(datapoints_in_cluster))

        return mean

    def get_full_clusters_of_data(self, clustered_datapoints, selected_features, data_to_cluster):
        data = self.data_munge(selected_features, data_to_cluster)
        num_features = len(selected_features)

        model_clusters = self.create_clusters(len(clustered_datapoints))

        # Calculate means for fisher score
        for index_of_cluster_of_datapoints in range(len(clustered_datapoints)):
            cluster_of_datapoints = clustered_datapoints[index_of_cluster_of_datapoints]
            model_clusters[index_of_cluster_of_datapoints] = cluster_of_datapoints

        means = [[0] * num_features] * self.k
        means = self.calculate_means(model_clusters, means)

        # Cluster all of the datapoints based on means
        full_clusters = self.create_clusters(len(means))
        for datum_index in range(len(data)):
            datum = data[datum_index]
            cluster_index = self.argmin_cluster(datum, means)
            full_clusters[cluster_index].append(data_to_cluster[datum_index])

        return full_clusters
    # </editor-fold>

    def agglomerate(self, distance_matrix, clusters):
        argmin_item = self.matrix_argmin(distance_matrix)
        i = argmin_item[0]
        j = argmin_item[1]

        clusters = self.adjust_clusters(clusters, i, j)

        # print("Min item: {}".format(distance_matrix[argmin_item[0]][argmin_item[1]]))
        # print("argminMin item: {}".format(argmin_item))

        ith_row = distance_matrix[i]
        ith_col = distance_matrix[:, i]

        jth_row = distance_matrix[j]
        jth_col = distance_matrix[:, j]

        ith_distances = ith_row + ith_col
        jth_distances = jth_row + jth_col

        distance_matrix = self.delete_closest_points_from_matrix(distance_matrix, i, j)
        distance_matrix = self.add_cluster_to_matrix(distance_matrix)
        distance_matrix = self.calculate_new_cluster_distances(distance_matrix, i, j, ith_distances, jth_distances)

        return distance_matrix, clusters

    def adjust_clusters(self, clusters, i, j):
        points_in_i = clusters[i]
        points_in_j = clusters[j]
        new_cluster_points = points_in_i + points_in_j

        del clusters[i]
        if i > j:
            del clusters[j]
        else:
            del clusters[j-1]

        clusters.append(new_cluster_points)
        return clusters


    def delete_closest_points_from_matrix(self, distance_matrix, i, j):
        distance_matrix = np.delete(distance_matrix, i, axis=0)
        distance_matrix = np.delete(distance_matrix, i, axis=1)

        if i > j:
            distance_matrix = np.delete(distance_matrix, j, axis=0)
            distance_matrix = np.delete(distance_matrix, j, axis=1)
        else:
            distance_matrix = np.delete(distance_matrix, j - 1, axis=0)
            distance_matrix = np.delete(distance_matrix, j - 1, axis=1)

        return distance_matrix

    def add_cluster_to_matrix(self, distance_matrix):
        matrix_length = len(distance_matrix)
        new_row = np.zeros(matrix_length)
        new_col = np.zeros(matrix_length+1)
        distance_matrix = np.insert(distance_matrix, matrix_length, new_row, axis=0)
        distance_matrix = np.insert(distance_matrix, matrix_length, new_col, axis=1)

        return distance_matrix

    def calculate_new_cluster_distances(self, distance_matrix, i, j, ith_distances, jth_distances):
        ith_distances = np.delete(ith_distances, i)
        jth_distances = np.delete(jth_distances, i)

        if i > j:
            ith_distances = np.delete(ith_distances, j)
            jth_distances = np.delete(jth_distances, j)
        else:
            ith_distances = np.delete(ith_distances, j-1)
            jth_distances = np.delete(jth_distances, j-1)

        matrix_length = len(distance_matrix)
        for index in range(matrix_length-1):
            ith_distance = ith_distances[index]
            jth_distance = jth_distances[index]

            distance_matrix[index][matrix_length-1] = min(ith_distance, jth_distance)

        return distance_matrix



    def matrix_argmin(self, distance_matrix):
        min_value = sys.maxsize
        min_ith = -1
        min_jth = -1
        for i in range(len(distance_matrix)-1):
            for j in range(i+1, len(distance_matrix)):
                value = distance_matrix[i][j]
                if value < min_value:
                    min_value = distance_matrix[i][j]
                    min_ith = i
                    min_jth = j
        return min_ith, min_jth


    def compute_initial_distances(self, distance_matrix, data):
        for datapoint_index_i in range(len(data)):
            for datapoint_index_j in range(datapoint_index_i, len(data)):
                distance = self.distance(data[datapoint_index_i], data[datapoint_index_j])
                distance_matrix[datapoint_index_i][datapoint_index_j] = distance
        return distance_matrix


    # <editor-fold desc="Helpers">
    def create_datapoint_clusters(self, clusters_with_ids, data):
        datapoint_clusters = []

        for cluster_index in range(len(clusters_with_ids)):
            cluster_with_ids = clusters_with_ids[cluster_index]

            datapoint_clusters.append([])
            for id in cluster_with_ids:
                datapoint_clusters[cluster_index].append(data[id])

        return datapoint_clusters

    def init_clusters(self, num_data_points):
        clusters = []
        for index in range(num_data_points):
            clusters.append([index])
        return clusters

    def distance(self, datapoint, mean):
        running_sum = 0
        for feature_value, feature_value_mean in zip(datapoint, mean):
            running_sum += (feature_value - feature_value_mean)**2

        return math.sqrt(running_sum)

    def create_clusters(self, number_of_clusters):
        cluster = {}
        for cluster_number in range(number_of_clusters):
            cluster[cluster_number] = []
        return cluster

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

    def data_munge(self, selected_features, data):
        new_data = []
        for data_point in data:
            new_data_point = []
            for selected_feature in selected_features:
                new_data_point.append(data_point[selected_feature])
            new_data.append(new_data_point)
        return new_data

    # </editor-fold>


# <editor-fold desc="Tests">
# all_data = CustomCSVReader.read_file("data/iris.data.txt", float)
# hac = HAC(3)
#
# data_id_clusters = hac.learn([2], all_data)

# trimmed_data = hac.data_munge([2], all_data)
#
# clusters = [[]]*3
# for cluster_index in range(len(data_id_clusters)):
#     cluster = data_id_clusters[cluster_index]
#     for id in cluster:
#         clusters[cluster_index] = clusters[cluster_index] + trimmed_data[id]
#
#
# plt.plot(clusters[0], np.zeros_like(clusters[0]), 'x', color='red')
# plt.plot(clusters[1], np.zeros_like(clusters[1]), 'x', color='blue')
# plt.plot(clusters[2], np.zeros_like(clusters[2]), 'x', color='green')
# plt.show()

# score = hac.evaluate(data_id_clusters, [2], all_data)
#
# print("SCORE")
# print(score)
# # </editor-fold>
#
#
# all_data = CustomCSVReader.read_file("data/glass.data.txt", float)
# hac = HAC(6)
#
# data_id_clusters = hac.learn([2], all_data)
#
# score = hac.evaluate(data_id_clusters, [2], all_data)
#
# print("SCORE")
# print(score)


#####

# all_data = CustomCSVReader.read_file("data/spambase.data.txt", float)
# hac = HAC(2)
#
# random.shuffle(all_data)
# training_data = all_data[: int(len(all_data)/10)]
#
# data_id_clusters = hac.learn([2], training_data)
# print("NOW EVALUATING")
#
# score = hac.evaluate(data_id_clusters, [2], all_data)
#
# print("SCORE")
# print(score)
