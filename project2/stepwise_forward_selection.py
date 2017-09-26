""" Created by Max Robinson 9/20/2017 """
import sys
import copy
import pprint
from testLearner import TestLeaner
from k_means import KMeans
from hierarchical_agglomerative_clustering import HAC
from customCsvReader import CustomCSVReader


class SFS:

    @staticmethod
    def select_features(Features, data_training, data_test, learner):
        """

        :param Features: an Array of unique integers that correspond to a feature number
        in a data point in the data_training and data_test data sets.
        :param data_training: a list of lists where each inner list is a data point with n features.
        :param data_test: a list of lists where each inner list is a data point with n features.
        :param learner: A class that has the two functions learn and evaluate
        :return: a list of which features were selected which optimized the learners score.
        """
        original_features = copy.deepcopy(Features)
        selected_features = []
        base_preference_f = -sys.maxsize

        while len(Features) >= 0:
            best_perf = -sys.maxsize
            best_feature = None

            for feature in Features:
                selected_features.append(feature)
                model = learner.learn(selected_features, data_training)
                curr_perf = learner.evaluate(model, selected_features, data_test)
                # print("Selected Features: {}".format(selected_features))
                # print("Score: {}".format(curr_perf))


                if curr_perf > best_perf:
                    best_perf = curr_perf
                    best_feature = feature

                selected_features.remove(feature)

            if best_perf > base_preference_f:
                base_preference_f = best_perf
                Features.remove(best_feature)
                selected_features.append(best_feature)
                # print()
                # print("Setting BEST selected Features")
                # print("Best Selected Features: {}".format(selected_features))
                # print("Score: {}".format(curr_perf))
                # print()


            else:
                break
        return selected_features, base_preference_f
        # return selected_features


# all_data = CustomCSVReader.read_file("data/iris.data.txt", float)
# feature_length = len(all_data[0]) - 1
#
# kLearner = KMeans(3)
# Features = list(range(feature_length))
# best_features = SFS.select_features(Features, all_data, all_data, kLearner)
#
# print("\nBest Features Selected: {}\n".format(best_features))

def run_kmeans_experiment(data_set_path, number_of_clusters, learner, fraction_of_data_used=1, data_type=float):
    print("Running {0} Experiment with k clusters = {1}".format(data_set_path, number_of_clusters))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)
    feature_selection_data = all_data[:int(len(all_data)/fraction_of_data_used)]
    feature_length = len(all_data[0]) - 1

    Features = list(range(feature_length))
    best_features = SFS.select_features(Features, feature_selection_data, all_data, learner)

    means = learner.learn(best_features[0], all_data)
    data_clusters = learner.get_clusters_for_means(means, best_features[0], all_data)


    print("The Final Selected Features are: (features are zero indexed) ")
    print("{}\n".format(best_features[0]))
    print("The Fisher Score for the clustering is: ")
    print("{}\n".format(best_features[1]))

    pp = pprint.PrettyPrinter(indent=2, width=400)
    print("For Clustered points, the key in the dictionary represents the cluster each data point belongs to. ")
    print("Clustered points: ")
    pp.pprint(data_clusters)


def run_hac_experiment(data_set_path, number_of_clusters, hac, fraction_of_data_used=1, data_type=float):
    print("Running {0} Experiment with k clusters = {1}".format(data_set_path, number_of_clusters))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)
    feature_selection_data = all_data[:int(len(all_data)/fraction_of_data_used)]
    feature_length = len(all_data[0]) - 1

    Features = list(range(feature_length))
    best_features = SFS.select_features(Features, feature_selection_data, all_data, hac)

    clusters_of_datapoint_ids = hac.learn(best_features[0], feature_selection_data)
    full_clusters = hac.get_full_clusters_of_data(clusters_of_datapoint_ids, best_features[0], all_data)



    print("The Final Selected Features are: (features are zero indexed) ")
    print("{}\n".format(best_features[0]))
    print("The Fisher Score for the clustering is: ")
    print("{}\n".format(best_features[1]))

    pp = pprint.PrettyPrinter(indent=2, width=400)
    print("For Clustered points, the key in the dictionary represents the cluster each data point belongs to. ")
    print("Clustered points: ")
    pp.pprint(full_clusters)



# KMeans experiments
sys.stdout = open('results/SFS-Kmeans-iris-results.txt', 'w')
run_kmeans_experiment("data/iris.data.txt", 3, KMeans(3))

sys.stdout = open('results/SFS-Kmeans-glass-results.txt', 'w')
run_kmeans_experiment("data/glass.data.txt", 6, KMeans(6))

sys.stdout = open('results/SFS-Kmeans-spambase-results.txt', 'w')
run_kmeans_experiment("data/spambase.data.txt", 2, KMeans(2))


# HAC experiments
sys.stdout = open('results/SFS-HAC-iris-results.txt', 'w')
run_hac_experiment("data/iris.data.txt", 3, HAC(3))

sys.stdout = open('results/SFS-HAC-glass-results.txt', 'w')
run_hac_experiment("data/glass.data.txt", 6, HAC(6))
#
# sys.stdout = open('results/SFS-HAC-spambase-results.txt', 'w')
# run_kmeans_experiment("data/spambase.data.txt", 2, HAC(2), fraction_of_data_used=10)



"""
Pseudo code

function SFS(Features, D_train, D_valid, Learn()): 
    F_0 = <>
    basePerf = -inf
    do:
        bestPerf = - inf
        for all Features in FeatureSpace do: 
            F_0 = F_0 + F
            h = Learn(F_0, D_train)
            currPerf = Perf(h, D_valid)
            if currPerf > bestPerf then:
                bestPerf = currPerf
                bestF = F
            end if
            F_0 = F_0 - F
        end for
        if bestPerf > basePerf then 
            basePerf = bestPerf
            F = F - bestF 
            F_0 = F_0 + bestF
        else
            exit (Break)
        end if
    until F = <> (is empty)
    return F_0

"""