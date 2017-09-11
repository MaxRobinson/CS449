from __future__ import division
import sys, csv, math, copy, random, pprint


# <editor-fold desc="Init Data">
def get_possible_domains():
    return [0, 1]

"""
return the label that is the positive label for the algorithm
"""
def get_positive_label():
    return 1

"""
return the label that is the negative label for the algorithm
"""
def get_negative_label():
    return 0

""" 
Returns the classification label for a given single record.
By conventions, the class label is stored as the last element 
the feature vector. 

input: 
+ record: list (a feature vector with a class label) 
"""
def get_class(record):
    return record[-1]


"""
A helper function to create a dictionary that holds the Naive Bayes Classifier distibutions 
for all of the  P(ai|ci)P(ai|ci)  probabilities, for each  A  where  A  is all attributes 
and ai  is a domain for a specific attribute.
The dictionary has the following strucutre:
{
    (feature#, Feature domain, 'label', label_value) : value
}

input:
+ None

return:
+ a dictionary with the structure specified in the above discription.
"""
def create_distribution_dict(number_features):
    domains = get_possible_domains()

    distribution = {}

    for feature_number in range(number_features):
        for domain in domains:
            distribution[(feature_number, domain, 'label', get_positive_label())] = 1
            distribution[(feature_number, domain, 'label', get_negative_label())] = 1

    return distribution


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

# <editor-fold desc="Learn">
"""
A helper function the key needed to access a given probability in the Naive Bayes Distribution dictionary, 
described in create_distribution_dict.

input:
attribute: a String that specifies the attribute for the probability to access
domain: a string that specifies the domain value for the probability to access
label_value: a string that specifies which classification label to use when accessing the probability.

output: 
a tuple with the structure: (feature#, domain_value, 'label', label_value)

"""
def create_distribution_key(feature_number, domain_value, label_value):
    return (feature_number, domain_value, 'label', label_value)

"""
A helper function to increment the count by 1, in the distribution dictionary, of a given key.

Used when counting the number of occurenses of a particular  A=ai,C=ciA=ai,C=ci  
when building out the distribution of the training set.

input:
+ distribution: a dictionary with the structure specified by create_distribution_dict
+ attribute: a String that specifies the attribute for the probability to access
+ domain: a string that specifies the domain value for the probability to access
+ label_value: a string that specifies which classification label to use when accessing the probability.

"""
def put_value_in_distribution(distribution, feature_number, domain_value, label_value):
    key = create_distribution_key(feature_number, domain_value, label_value)
    distribution[key] += 1

"""
A helper function that returns the number of records that have a given label.

input:
records: a list of records.

return:
count: the number of records with the specified label
"""
def get_label_count(records, label):
    count = 0
    for record in records:
        if get_class(record) == label:
            count += 1
    return count


"""
input:
pos_count: an int, the number of records with the "positive" label in the training set.
neg_count: an int, the number of records with the "negative" label in the training set.
distribution: a dictionary, with the structure specified in create_distribution_dict

return:
+ distribution: a dictionary, with the structure specified in create_distribution_dict, 
now with values that are probabilites rather than raw counts. Probability is calculated 
according to the above formula.
"""
def create_percentages(pos_count, neg_count, distribution):
    pos_count_plus_1 = pos_count + 1
    neg_count_plus_1 = neg_count + 1

    pos_label = get_positive_label()
    neg_label = get_negative_label()

    for key in distribution:
        if key[3] == pos_label:
            distribution[key] = distribution[key] / pos_count_plus_1
        elif key[3] == neg_label:
            distribution[key] = distribution[key] / neg_count_plus_1

    return distribution


"""
The main function that learns the distribution for the Naive Bayes Classifier.
The function works as follows:
Create initial distribution counts
get positive label counts
get negative label counts

for each record in the training set:

For each attribute, and domain_value for the attribute:
    put the value into the distribution (i.e incriment the value for that attribute, domain, and label tuple
        the Corresponding value in the distribution is (Attribute, domain_value, 'label', actual label for record)
    change the distribution from counts to probabilities
  add special entries in the distribution for the Probability of each possible label.
    the Probability of a given label is as follows:  
      P(ci)=Num(ci)/SizeOfTrainingSetP(ci)=Num(ci)SizeOfTrainingSet 
We then return the learned distribution, as our Naive Bayes Classifier.

input:
records: a list of records, as described by the create_record function.

return:
distribution: a dictionary, with the structure specified in create_distribution_dict, 
with values that are the probabilites for each  A  and  C  so that we have  P(A=ai|C=ci)
"""
def learn(records):
    distribution = create_distribution_dict(len(records[0])-1)
    pos_count = get_label_count(records, get_positive_label())
    neg_count = get_label_count(records, get_negative_label())

    for record in records:
        for feature_index in range(len(record)-1):
            feature_value = record[feature_index]
            put_value_in_distribution(distribution, feature_index, feature_value, get_class(record))

    distribution = create_percentages(pos_count, neg_count, distribution)
    distribution[('label', get_positive_label())] = pos_count / (pos_count + neg_count)
    distribution[('label', get_negative_label())] = neg_count / (pos_count + neg_count)

    return distribution


# </editor-fold>

# <editor-fold desc="Classify">
"""
A helper function that calculates the un_normalized probability of a given instance (record), for a given label.

input:
+ distribution: a dictionary, with the structure specified in create_distribution_dict, with values that are the probabilites.
+ instance: a record, as described by create_record
+ label: a string that describes a given label value.

return:
+ un_normalized_prob: a float that represents the un_normalized probability that a record belongs 
to the given class label.
"""
def calculate_probability_of(distribution, instance, label):
    un_normalized_prob = distribution[('label', label)]
    for feature_index in range(len(instance)-1):
        feature_value = instance[feature_index]
        key = create_distribution_key(feature_index, feature_value, label)
        un_normalized_prob *= distribution[key]

    return un_normalized_prob


"""
A helper function that normalizes a list of probabilities. The list of probabilities is for a single record, and should have the following structure:
[(label, probability), (label, probability)]

These probabilities should be un_normalized probabilities for each label.
This function normalizes the probabilities by summing the probabilities for each label together, then calculating the normalized probability for each label by dividing the probability for that label by the sum of all the probabilities.
This normalized probability is then placed into a new list with the same structure and same corresponding label.
The list of normalized probabilies is then SORTED in descending order. I.E. the label with the most likely possibility is in index position 0 for the list of probabilities**
This new normalized list of probabilities is then returned.

input:
probability_list: a list of tuples, as described by: [(label, probability), (label, probability)]

return:
normalized_list: a list of tuples, as described by: [(label, probability), (label, probability)] with the probabilities being normalized as described above.
"""
def normalize(probability_list):
    sum_of_probabilities = 0

    normalized_list = []

    for prob_tuple in probability_list:
        sum_of_probabilities += prob_tuple[1]

    for prob_tuple in probability_list:
        normalized_prob = prob_tuple[1] / sum_of_probabilities

        normalized_list.append((prob_tuple[0], normalized_prob))

    normalized_list.sort(key=lambda x: x[1], reverse=True)
    return normalized_list


"""
A helper that does most of the work to classifiy a given instance of a record.
It works as follows:

create a list of possible labels
initialize results list.
for each label
    calculate the un_normalized probability of the instance using calculate_probabily_of
    add the probability to the results list as a tuple of (label, un_normalized probability)
normalize the probabilities, using normalize
    note that now the list of results (a list of tuples) is now sorted in descending order by the value of the probability
return the normalized probabilities for that instance of a record.

input:
+ distribution: a dictionary, with the structure specified in create_distribution_dict, with values that are the probabilites.
+ instace: a record, as described by create_record


return:
probability_results: a List of tuples with the structure as: [(label, normalized probability), (label, normalized probability)] 
sorted in descending order by probability.
"""
def classify_instance(distribution, instance):
    labels = [get_positive_label(), get_negative_label()]

    probability_results = []

    for label in labels:
        probability = calculate_probability_of(distribution, instance, label)
        probability_results.append((label, probability))

    probability_results = normalize(probability_results)

    return probability_results

"""
A function to classify a list of instances(records).
Given a list of instances (records), classify each instance using classify_instance and put the result into a result list. Return the result list after each instance has been classified.
The Structure of the return list will be a List of lists where each inner list is a list of tuples, as described by the classify_instance function. An example will look as follows:
[ [('e', .999),('p', .001)], [('p', .78), ('e', .22)] ]

input:
+ distribution: a dictionary, with the structure specified in create_distribution_dict, with values that are the probabilites.
+ instace: a record, as described by create_record

return:
+ results: a list of lists of tuples as described above.
"""
def classify(distribution, instances):
    results = []
    for instance in instances:
        results.append(classify_instance(distribution, instance))

    return results

# </editor-fold>

# <editor-fold desc="Evaluate">
"""
The main evaluation method. Uses a simple  NumErrors / totalDataPoints  
to calculate the error rate of the Naive Bayes Classifier.

Given a list of records (test_data) and a list of predicted classifications for that data set, run through both lists, 
and compire the label for the record to the predicted classification. If they do not match, increase the 
number of errors seen.

Return the number of erros seen divided by the total number of data points. This is the error rate.

input:
test_data: a list of records
classifications: a list of lists of tuples, as described by the classify function.

return:
error_rate: a float that represents the number of errors / total number of data points.
"""
def evaluate(test_data, classifications):
    number_of_errors = 0
    for record, classification in zip(test_data, classifications):
        if get_class(record) != classification[0][0]:
            number_of_errors += 1

    return number_of_errors/len(test_data)
# </editor-fold>

# <editor-fold desc="Data Pre-process">

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
    test_records = read_file(data_set_path)
    test_records = pre_process(test_records, positive_class_name)

    random.shuffle(test_records)

    two_thirds = 2 * int(math.floor(len(test_records)/3))
    set_1 = test_records[:two_thirds]
    set_2 = test_records[two_thirds:]

    distro_1 = learn(set_1)

    pp = pprint.PrettyPrinter(indent=2)
    print("Learned Naive Bayes Distribution: ")
    print("Keys are structured as follows: (feature#, possible domain values 0 or 1, 'label', label value)")
    print("Special Key's that are ('label', possible_class_value) are the percentage of the distribution with that class label")
    pp.pprint(distro_1)
    print()

    # Evaluate
    c2 = classify(distro_1, set_2)

    print("Results for Test Set: \n")
    for predicted_class, test_record in zip(c2, set_2):
        print("Predicted Class: {}".format(predicted_class[0][0]))
        print("Actual Class: {}".format(get_class(test_record)))
        print("Test feature Vector (last feature is actual class): \n{} \n".format(test_record))

    evaluation_1 = evaluate(set_2, c2)
    print("Error Rate = {}".format(evaluation_1))
    print()
# </editor-fold>

"""
This is the main method for the experiments. 

These calls run the experiments on the different data sets with the appropriate 
positive class to be used. 

In addition, standard out is piped to a file so that all logging statements are 
captured in their own unique file per experiment. 
"""
sys.stdout = open('results/NB-Bresat-Cancer-results.txt', 'w')
run_experiment("data/breast-cancer-wisconsin.data.new.txt", 1)
run_experiment("data/breast-cancer-wisconsin.data.new.txt", 0)

sys.stdout = open('results/NB-soybean-small-results.txt', 'w')
run_experiment("data/soybean-small.data.new.txt", "D1")
run_experiment("data/soybean-small.data.new.txt", "D2")
run_experiment("data/soybean-small.data.new.txt", "D3")
run_experiment("data/soybean-small.data.new.txt", "D4")

sys.stdout = open('results/NB-house-votes-84-results.txt', 'w')
run_experiment("data/house-votes-84.data.new.txt", "democrat")
run_experiment("data/house-votes-84.data.new.txt", "republican")

sys.stdout = open('results/NB-iris-results.txt', 'w')
run_experiment("data/iris.data.new.txt", "Iris-setosa")
run_experiment("data/iris.data.new.txt", "Iris-versicolor")
run_experiment("data/iris.data.new.txt", "Iris-virginica")

sys.stdout = open('results/NB-glass-results.txt', 'w')
run_experiment("data/glass.data.new.txt", 1)
run_experiment("data/glass.data.new.txt", 2)
run_experiment("data/glass.data.new.txt", 3)
run_experiment("data/glass.data.new.txt", 5)
run_experiment("data/glass.data.new.txt", 6)
run_experiment("data/glass.data.new.txt", 7)
