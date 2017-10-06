""" Created by Max 10/4/2017 """
import sys
import pprint

from customCsvReader import CustomCSVReader
from CrossValidation import CrossValidation
from knn import Knn
# import CondensedKnn


# <editor-fold desc="Experiment">
def run_experiment(data_set_path, k_nearest, learner, classification, fraction_of_data_used=1, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, trainnig and test data.

    Creates the GA and pass the needed data to it. It returns a list of selected features.
    The mean is then retrieved for those features, and we cluster all of the data based on the means

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with k nearest = {1}".format(data_set_path, k_nearest))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)
    # feature_selection_data = all_data[:int(len(all_data)/fraction_of_data_used)]
    # feature_length = len(all_data[0]) - 1
    # Features = list(range(feature_length))

    # For algorithms knn Cross Validation
    cv = CrossValidation(5, learner, classification)
    average_mse = cv.cross_validation_regression(all_data)

    print("Average MSE: {}".format(average_mse[0]))
    print("Standard Deviation: {}".format(average_mse[1]))
    print(average_mse)

    # print("The Final Selected Features are: (features are zero indexed) ")
    # print("{}\n".format(selected_features))
    # print("The Fisher Score for the clustering is: ")
    # print("{}\n".format(best_features["evaluation"]))
    #
    # pp = pprint.PrettyPrinter(indent=2, width=400)
    # print("For Clustered points, the key in the dictionary represents the cluster each data point belongs to. ")
    # print("Clustered points: ")
    # pp.pprint(data_clusters)

def run_classification_experiment(data_set_path, k_nearest, learner, classification, fraction_of_data_used=1, data_type=float):
    print("Running {0} Experiment with k nearest = {1}".format(data_set_path, k_nearest))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)

    cv = CrossValidation(5, learner, classification)
    average_error_rate = cv.cross_validation_classification(all_data)

    print("Average Error Rate: {}".format(average_error_rate[0]))
    print("Standard Deviation: {}".format(average_error_rate[1]))
    print(average_error_rate)

# </editor-fold>



# KMeans experiments
# sys.stdout = open('results/SFS-Kmeans-iris-results.txt', 'w')
run_experiment("data/machine.data.new.txt", 2, Knn(2), False)
print()

# run_classification_experiment("data/ecoli.data.new.txt", 6, Knn(6), False)