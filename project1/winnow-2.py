from __future__ import division

import csv
import math
import random
import sys
import pprint


# <editor-fold desc="Init Data">
"""
A helper function to read in the data from a CSV file, and transform it into a list of feature vectors

input:
+ path (optional): the path to the csv file you wish to read in.
return:
+ records: A list of records. Records have the shape described by the create_record description.
"""
def read_file(path=None):
    if path is None:
        path = 'data/breast-cancer-wisconsin.data.txt'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    for inner_list in csv_list:
        new_values = []
        for feature in inner_list:
            try:
                new_feature = int(feature)
                new_values.append(new_feature)
            except Exception:
                new_values.append(feature)

        inner_list[:] = new_values

    return csv_list
# </editor-fold>


# <editor-fold desc="Helpers">
""" 
Returns the classification label for a given single record.
By conventions, the class label is stored as the last element 
the feature vector. 

input: 
+ record: list (a feature vector with a class label) 
"""
def get_class(record):
    return record[-1]
# </editor-fold>


# <editor-fold desc="Winnow-2">
"""
Initializes the weights for the Winnow-2 algorithm. 
uses an initial weight of 1 for all weights. 

input: 
record_length: length of the feature vector (without the class value) 

output: 
list of 1's 
"""
def init_weights(record_length):
    return [1]*record_length


"""
returns the prediction of the winnow-2 model for a given record and model and threshold. 

Uses the Sum of the wieghts * feature value. 
If the sum is larger than the threshold, return 1, else return 0

input: 
record: list of binary features
weights: list of floats
threshold: float
"""
def calculate_model_prediction(record, weights, threshold):
    running_sum = 0
    for feature, weight in zip(record, weights):
        running_sum += feature * weight
    if running_sum >= threshold:
        return 1
    else:
        return 0

"""
Given a record and the weights, a learning rate alpha, and if we should promote or demote,
update the wieghts of the model. 
if promote = True, use the promote function for winnow-2. If False, use demote. 

input: 
record: list of binary features.
weights: list of floats, the model 
alpha: float, the learning rate.
promote: boolean. Says to promote or demote.  

return: 
new_wieghts: list of floats, new model. 
"""
def update_weights(record, weights, alpha, promote=True):
    new_weights = []
    for feature, weight in zip(record, weights):
        if feature == 1:
            if promote:
                new_weights.append(alpha * weight)
            else:
                new_weights.append(weight / alpha)
        else:
            new_weights.append(weight)

    return new_weights


"""
The main method behind the winnow-2 algorithm.
for each record in the training set, make a prediction. 
If that prediction is correct, move on. Else update the weights. 

When all done, return the model 

input: 
training_set: list of feature vectors with classes
theta: float, the threshold for the winnow algorithm
learning_rate: float, the rate at which the model should update. 
"""
def winnow_2(training_set, theta, learning_rate):
    weights = init_weights(len(training_set[0]) - 1)
    threshold = theta
    alpha = learning_rate
    random.shuffle(training_set)

    count = 0
    # print("Iteration #{0} (Initial Weights) : {1}".format(count, weights))
    for record in training_set:
        prediction = calculate_model_prediction(record, weights, threshold)
        actual_class = get_class(record)
        if prediction != actual_class:
            if prediction < actual_class:
                weights = update_weights(record, weights, alpha, True)
            else:
                weights = update_weights(record, weights, alpha, False)
        count += 1
        # print("Iteration #{0} : {1}".format(count, weights))

    return weights
# </editor-fold>


# <editor-fold desc="Verification">
"""
A helper method to evaluate the winnow-2 model with a test set. 

input: 
model: list of floats for the winnow algorithm
test: the test data, list of feature vectors
theshold: the threshold used for the winnow-2 predictions. 

return: 
A tuple of the (error_rate, the predictions for each test feature vector, the model) 

"""
def evaluate_winnow_2(model, test, threshold):
    results = []
    error = 0
    for record in test:
        prediction = calculate_model_prediction(record, model, threshold)
        results.append(prediction)
        if prediction != get_class(record):
            error += 1

    error_rate = error/len(test)
    return (error_rate, results, model)

# </editor-fold>

# <editor-fold desc="preprocess">
"""
Pre-processes the data for a given test run. 

The data is preprocessed by taking a positive class label, and modifying the in memory data to replace the 
positive_class_name with a 1, and all other classification names as 0, negative. 

This allows for easier binary classificaiton 

input: 
+ data: list of feature vecotrs
+ positive_class_name: Stirng, class to be the positive set.

"""
def pre_process(data, positive_class_name):
    new_data = []
    for record in data:
        current_class = get_class(record)
        if current_class == positive_class_name:
            record[-1] = 1
        else:
            record[-1] = 0
        new_data.append(record)
    return new_data
# </editor-fold>

# <editor-fold desc="Experiment">
"""
The main work horse for running the experiments and output the approriate information into a file

Works by reading in the data, setting up the 67/33 split of trainnig and test data. 
Next it creates the Naive bayes distribution, using the "Learn" function. 
It logs the distribution in a readable manner. 

Finally it classifies the test set. 
It prints all information needed in a human readable way. 

Finally we print the error rate for the test set. 
"""
def run_experiment(data_set_path, positive_class_name):
    print("Running {0} Experiment with positive class={1}".format(data_set_path, positive_class_name))
    records = read_file(data_set_path)
    records = pre_process(records, positive_class_name)
    # results = test_winnow_2(test_records)

    random.shuffle(records)

    one_third_data_length = int(math.floor(len(records)/3))
    training = records[:2*one_third_data_length]
    test = records[2*one_third_data_length:]

    threshold = .75

    model = winnow_2(training, .75, 2)

    pp = pprint.PrettyPrinter(indent=2)
    print("Learned Winnow-2 Model: ")
    pp.pprint(model)

    results = evaluate_winnow_2(model, test, threshold)
    # print("Results: \n model: {0} \n classifications on test set: {1}".format(results[2], results[1]))
    print("Results for Test Set: \n")
    for predicted_class, test_record in zip(results[1], test):
        print("Predicted Class: {}".format(predicted_class))
        print("Actual Class: {}".format(get_class(test_record)))
        print("Test feature Vector (last feature is actual class): \n{} \n".format(test_record))
    print("Error Rate for entire test set = {} \n".format(results[0]))
    print()



# </editor-fold>

"""
This is the main method for the experiments. 

These calls run the experiments on the different data sets with the appropriate 
positive class to be used. 

In addition, standard out is piped to a file so that all logging statements are 
captured in their own unique file per experiment. 
"""

sys.stdout = open('results/Winnow-Bresat-Cancer-results.txt', 'w')

run_experiment("data/breast-cancer-wisconsin.data.new.txt", 1)
run_experiment("data/breast-cancer-wisconsin.data.new.txt", 0)
#
sys.stdout = open('results/Winnow-soybean-small-results.txt', 'w')
run_experiment("data/soybean-small.data.new.txt", "D1")
run_experiment("data/soybean-small.data.new.txt", "D2")
run_experiment("data/soybean-small.data.new.txt", "D3")
run_experiment("data/soybean-small.data.new.txt", "D4")

sys.stdout = open('results/Winnow-house-votes-84-results.txt', 'w')
run_experiment("data/house-votes-84.data.new.txt", "democrat")
run_experiment("data/house-votes-84.data.new.txt", "republican")

sys.stdout = open('results/Winnow-iris-results.txt', 'w')
run_experiment("data/iris.data.new.txt", "Iris-setosa")
run_experiment("data/iris.data.new.txt", "Iris-versicolor")
run_experiment("data/iris.data.new.txt", "Iris-virginica")

sys.stdout = open('results/Winnow-glass-results.txt', 'w')
run_experiment("data/glass.data.new.txt", 1)
run_experiment("data/glass.data.new.txt", 2)
run_experiment("data/glass.data.new.txt", 3)
run_experiment("data/glass.data.new.txt", 5)
run_experiment("data/glass.data.new.txt", 6)
run_experiment("data/glass.data.new.txt", 7)


