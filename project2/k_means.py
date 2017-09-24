""" Created by Max 9/22/2017 """
import random


class KMeans:

    def __init__(self, k):
        self.k = k


    def learn(self, selected_features, data_training, mu=.001):
        data = self.data_munge(selected_features, data_training)

        means = self.init_k_means(data_training, self.k)
        clusters = [[]]

        previous_means = None
        while not self.has_stopped_changing(means, previous_means, mu):
            previous_means = means

            # for datapoint in data:
                





    def evaluate(self, model, selected_features, data_test):
        pass




    def data_munge(self, selected_features, data):
        new_data = []
        for data_point in data:
            new_data_point = []
            for selected_feature in selected_features:
                new_data_point.append(data_point[selected_feature])
            new_data.append(new_data_point)
        return new_data

    def has_stopped_changing(self, means, previous_means, mu):
        if previous_means is None:
            return False
        for mean, previous_mean in zip(means, previous_means):
            if not (mean - previous_mean <= mu):
                return False
        return True

    def init_k_means(self, data, num_k_s):
        means = []

        data_length = len(data)
        for x in range(num_k_s):
            random_data_index = random.randint(0, data_length)
            means.append(data[random_data_index])

        return means



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
