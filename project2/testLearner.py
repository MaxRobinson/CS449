""" Created by Max 9/23/2017 """
import random


class TestLeaner:

    def learn(self, selected_features, data_training):
        data = self.data_munge(selected_features, data_training)

        return [0, 0, 1, 1]

    def evaluate(self, model, selected_features, data_test):
        data = self.data_munge(selected_features, data_test)

        return random.randint(0, 1000)

    def data_munge(self, selected_features, data):
        new_data = []
        for data_point in data:
            new_data_point = []
            for selected_feature in selected_features:
                new_data_point.append(data_point[selected_feature])
            new_data.append(new_data_point)
        return new_data

