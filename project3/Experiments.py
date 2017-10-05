""" Created by Max 10/4/2017 """
import sys
import pprint

import CustomCSVReader
import CrossValidation
import Knn
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
    print("Running {0} Experiment with k clusters = {1}".format(data_set_path, k_nearest))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)
    # feature_selection_data = all_data[:int(len(all_data)/fraction_of_data_used)]
    # feature_length = len(all_data[0]) - 1
    # Features = list(range(feature_length))

    # For algorithms knn Cross Validation
    cv = CrossValidation(5, learner, classification)
    average_error = cv.cross_validation(all_data)

    print(average_error)

    # print("The Final Selected Features are: (features are zero indexed) ")
    # print("{}\n".format(selected_features))
    # print("The Fisher Score for the clustering is: ")
    # print("{}\n".format(best_features["evaluation"]))
    #
    # pp = pprint.PrettyPrinter(indent=2, width=400)
    # print("For Clustered points, the key in the dictionary represents the cluster each data point belongs to. ")
    # print("Clustered points: ")
    # pp.pprint(data_clusters)
# </editor-fold>



# KMeans experiments
# sys.stdout = open('results/SFS-Kmeans-iris-results.txt', 'w')
run_experiment("data/machine.data.new.txt", 3, Knn(3), False)