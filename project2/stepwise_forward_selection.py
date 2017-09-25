""" Created by Max Robinson 9/20/2017 """
import sys
import copy
import pprint
from testLearner import TestLeaner
from k_means import KMeans
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
                print("Selected Features: {}".format(selected_features))
                print("Score: {}".format(curr_perf))


                if curr_perf > best_perf:
                    best_perf = curr_perf
                    best_feature = feature

                selected_features.remove(feature)

            if best_perf > base_preference_f:
                base_preference_f = best_perf
                Features.remove(best_feature)
                selected_features.append(best_feature)
                print()
                print("Setting BEST selected Features")
                print("Best Selected Features: {}".format(selected_features))
                print("Score: {}".format(curr_perf))
                print()


            else:
                break
        return selected_features, base_preference_f


# all_data = CustomCSVReader.read_file("data/iris.data.txt", float)
# feature_length = len(all_data[0]) - 1
#
# kLearner = KMeans(3)
# Features = list(range(feature_length))
# best_features = SFS.select_features(Features, all_data, all_data, kLearner)
#
# print("\nBest Features Selected: {}\n".format(best_features))

def run_experiment(data_set_path, number_of_clusters, learner, data_type=float):
    print("Running {0} Experiment with k clusters ={1}".format(data_set_path, number_of_clusters))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)
    feature_length = len(all_data[0]) - 1

    Features = list(range(feature_length))
    best_features = SFS.select_features(Features, all_data, all_data, learner)

    print("The Final Selected Features are: ")
    print("{}".format(best_features[0]))
    print("The Fisher Score for the features is: ")
    print("{}".format(best_features[1]))

    # pp = pprint.PrettyPrinter(indent=2)
    # print("Learned Naive Bayes Distribution: ")
    # print("Keys are structured as follows: (feature#, possible domain values 0 or 1, 'label', label value)")
    # print("Special Key's that are ('label', possible_class_value) are the percentage of the distribution with that class label")
    # pp.pprint(distro_1)
    # print()
    #
    # # Evaluate
    # c2 = classify(distro_1, set_2)
    #
    # print("Results for Test Set: \n")
    # for predicted_class, test_record in zip(c2, set_2):
    #     print("Predicted Class: {}".format(predicted_class[0][0]))
    #     print("Actual Class: {}".format(get_class(test_record)))
    #     print("Test feature Vector (last feature is actual class): \n{} \n".format(test_record))
    #
    # evaluation_1 = evaluate(set_2, c2)
    # print("Error Rate = {}".format(evaluation_1))
    # print()


# run_experiment("data/iris.data.txt", 3, KMeans(3))
# run_experiment("data/glass.data.txt", 6, KMeans(6))
run_experiment("data/spambase.data.txt", 2, KMeans(2))

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