from __future__ import division
import sys, csv, math, copy, random, pprint


# <editor-fold desc="Init Data">
def get_possible_domains():
    return [0, 1]


def get_positive_label():
    return 1


def get_negative_label():
    return 0


def get_class(record):
    return record[-1]


def create_distribution_dict(number_features):
    domains = get_possible_domains()

    distribution = {}

    for feature_number in range(number_features):
        for domain in domains:
            distribution[(feature_number, domain, 'label', get_positive_label())] = 1
            distribution[(feature_number, domain, 'label', get_negative_label())] = 1

    return distribution


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
def create_distribution_key(feature_number, domain_value, label_value):
    return (feature_number, domain_value, 'label', label_value)


def put_value_in_distribution(distribution, feature_number, domain_value, label_value):
    key = create_distribution_key(feature_number, domain_value, label_value)
    distribution[key] += 1


def get_label_count(records, label):
    count = 0
    for record in records:
        if get_class(record) == label:
            count += 1
    return count


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

def calculate_probability_of(distribution, instance, label):
    un_normalized_prob = distribution[('label', label)]
    for feature_index in range(len(instance)-1):
        feature_value = instance[feature_index]
        key = create_distribution_key(feature_index, feature_value, label)
        un_normalized_prob *= distribution[key]

    return un_normalized_prob


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


def classify_instance(distribution, instance):
    labels = [get_positive_label(), get_negative_label()]

    probability_results = []

    for label in labels:
        probability = calculate_probability_of(distribution, instance, label)
        probability_results.append((label, probability))

    probability_results = normalize(probability_results)

    return probability_results


def classify(distribution, instances):
    results = []
    for instance in instances:
        results.append(classify_instance(distribution, instance))

    return results

# </editor-fold>

# <editor-fold desc="Evaluate">
def evaluate(test_data, classifications):
    number_of_errors = 0
    for record, classification in zip(test_data, classifications):
        if get_class(record) != classification[0][0]:
            number_of_errors += 1

    return number_of_errors/len(test_data)
# </editor-fold>

# <editor-fold desc="Data Pre-process">
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
