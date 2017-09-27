""" Created by Max 9/24/2017 """
import numpy as np


def fisher_score(mean_vectors, clustered_data):
    """
    Calculates the fisher score as described by the Fisher Score PDF for the project.

    NOTE: the W vector is always 1's because the data that is in each of the clusters is only has the selected features
    in it any ways, and the mean vector is only the length of the selected features. Thus we do not need to use the W
    vector to prune out data that doesn't matter. Instead we can use it to transform out data to a scalar.

    :param mean_vectors: list of mean vectors
    :param clustered_data: The data clustered into len(mean_vectors) clusters
    :return: float, Fisher Score
    """
    k_clusters = len(mean_vectors)
    number_of_features = len(mean_vectors[0])

    w = np.ones(number_of_features)
    w_t = w.reshape(number_of_features, 1)
    mean_of_mean_vector = create_mean_of_mean_vector(mean_vectors)


    sum_matrix = np.zeros((number_of_features, number_of_features))

    for mean_vector in mean_vectors:
        np_mean_vector = np.array(mean_vector)
        mean_diff = np_mean_vector - mean_of_mean_vector
        mean_diff_transpose = mean_diff.reshape(number_of_features, 1)

        outer_product = mean_diff * mean_diff_transpose
        sum_matrix = sum_matrix + outer_product

    numerator = w.dot(sum_matrix)
    numerator = numerator.dot(w_t)
    numerator_value = numerator[0]

    # denominator
    summed_covariance = np.zeros((number_of_features,number_of_features))
    for cluster_number, cluster_datapoints in clustered_data.items():
        kth_coveriance = covariance(cluster_datapoints, number_of_features, mean_vectors[cluster_number]) # Needs work
        kth_coveriance = np.array(kth_coveriance)
        summed_covariance = summed_covariance + kth_coveriance

    denominator = w.dot(summed_covariance)
    denominator = denominator.dot(w_t)
    denominator_value = denominator[0]
    denominator_value *= k_clusters


    SCORE = numerator_value / float(denominator_value)

    return SCORE


def covariance(datapoints, num_features,  mean_feature_vector_for_c):
    """
    Calculates the Covariance Matrix that is needed for each cluster of data.
    The covariance matrix is calculated as described by the Fisher Score PDF for the project
    :param datapoints: datapoints in cluster
    :param num_features: number of features per datapoint.
    :param mean_feature_vector_for_c: the mean feature vector for the cluster
    :return: The covariance matrix for the cluster.
    """
    cov_matrix = [[0]*num_features]*num_features
    for feature_number_i in range(num_features):
        for feature_number_j in range(num_features):
            running_sum = 0
            for datapoint in datapoints:
                ith_diff = datapoint[feature_number_i] - mean_feature_vector_for_c[feature_number_i]
                jth_diff = datapoint[feature_number_j] - mean_feature_vector_for_c[feature_number_j]
                running_sum += ith_diff * jth_diff

            if len(datapoints) <= 0:
                running_sum = 0
            else:
                running_sum = running_sum / float(len(datapoints))

            cov_matrix[feature_number_i][feature_number_j] = running_sum
    return cov_matrix


def create_mean_of_mean_vector(mean_vectors):
    """
    Helper function to create the mean of means vector for the Fisher score.
    :param mean_vectors: list of mean vectors.
    :return: mean of mean vector (list)
    """
    mean_vector_matrix = np.array(mean_vectors)
    vector_sums = np.sum(mean_vector_matrix, axis=0)
    mean_of_mean_vector = vector_sums / len(mean_vectors)
    return mean_of_mean_vector



######### Tests ##########

# test_mean_vectors = [[0,1,2,3,4], [0,1,2,3,4]]
# mm = create_mean_of_mean_vector(test_mean_vectors)
# print(mm)
