""" Created by Max 9/24/2017 """
import numpy as np


def fisher_score(mean_vectors, clustered_data):
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
    for cluster, cluster_datapoints in clustered_data.items():
        covariance(cluster_datapoints) # Needs work
        # TODO: Finish Implementation


def covariance(datapoints):
    pass
    # TODO: Finish Implementation


def create_mean_of_mean_vector(mean_vectors):
    mean_vector_matrix = np.array(mean_vectors)
    vector_sums = np.sum(mean_vector_matrix, axis=0)
    mean_of_mean_vector = vector_sums / len(mean_vectors)
    return mean_of_mean_vector



######### Tests ##########

test_mean_vectors = [[0,1,2,3,4], [0,1,2,3,4]]
mm = create_mean_of_mean_vector(test_mean_vectors)
print(mm)
