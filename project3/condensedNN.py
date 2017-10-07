""" Created by Max 10/6/2017 """
import random
import copy
from knn import Knn
from customCsvReader import CustomCSVReader

class CondensedNN:
    def __init__(self):
        self.knn = Knn(1, True)

    def condense(self, training_set):
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
        return self.knn.learn(z, datapoint)


# all_data = CustomCSVReader.read_file("data/ecoli.data.new.txt")
# random.shuffle(all_data)
# training_data = all_data[:2*(int(len(all_data)/3))]
#
# test_data = all_data[2*(int(len(all_data)/3)):]
#
#
# cnn = CondensedNN()
# z = cnn.condense(training_data)
#
# print(len(z))
# print(z)
# print("All Data len {}".format(len(all_data)))
#
# knn = Knn(4, True)
#
# results = knn.test(z, test_data)
#
# error = 0
# for result, datapoint in zip(results, test_data):
#     if result != datapoint[-1]:
#         error += 1
#
# print(error/len(results))


