""" Created by Max Robinson 9/20/2017 """
import sys
import copy
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

                if curr_perf > best_perf:
                    best_perf = curr_perf
                    best_feature = feature

                selected_features.remove(feature)

            if best_perf > base_preference_f:
                base_preference_f = best_perf
                Features.remove(best_feature)
                selected_features.append(best_feature)

            else:
                break
        return selected_features


all_data = CustomCSVReader.read_file("data/iris.data.txt", float)
# data_training = all_data[:2*int(len(all_data)/3)]
# data_test = all_data[2*int(len(all_data)/3):]
feature_length = len(all_data[0]) - 1

Features = list(range(feature_length))

# testLearner = TestLeaner()
kLearner = KMeans(3) # TODO: Finish Implementation

best_features = SFS.select_features(Features, all_data, all_data, kLearner)

print(best_features)

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