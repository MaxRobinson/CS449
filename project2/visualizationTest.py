""" Created by Max 9/24/2017 """
import numpy as np
import matplotlib.pyplot as plt
from customCsvReader import CustomCSVReader


def data_munge(selected_features, data):
    new_data = []
    for data_point in data:
        new_data_point = []
        for selected_feature in selected_features:
            new_data_point.append(data_point[selected_feature])
        new_data.append(new_data_point)
    return new_data


all_data = CustomCSVReader.read_file("data/iris.data.txt", float)

data = data_munge([2], all_data)
np.zeros_like(data)

plt.plot(data, np.zeros_like(data), 'x', color='red')
plt.show()

