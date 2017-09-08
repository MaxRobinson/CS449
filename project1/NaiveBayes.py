from __future__ import division
import csv, math, copy, random


# <editor-fold desc="Init Data">
def attributes_domains():
    return {
        'label': ['e', 'p', '?'],
        'cap-shape': ['b', 'c', 'x', 'f', 'k', 's', '?'],
        'cap-surface': ['f', 'g', 'y', 's', '?'],
        'cap-color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y', '?'],
        'bruises?': ['t', 'f', '?'],
        'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's', '?'],
        'gill-attachment': ['a', 'd', 'f', 'n', '?'],
        'gill-spacing': ['c', 'w', 'd', '?'],
        'gill-size': ['b', 'n', '?'],
        'gill-color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y', '?'],
        'stalk-shape': ['e', 't', '?'],
        'salk-root': ['b', 'c', 'u', 'e', 'z', 'r', '?'],
        'stalk-surface-above-ring': ['f', 'y', 'k', 's', '?'],
        'stalk-surface-below-ring': ['f', 'y', 'k', 's', '?'],
        'stalk-color-above-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y', '?'],
        'stalk-color-below-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y', '?'],
        'veil-type': ['p', 'u', '?'],
        'veil-color': ['n', 'o', 'w', 'y', '?'],
        'ring-number': ['n', 'o', 't', '?'],
        'ring-type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z', '?'],
        'spore-print-color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y', '?'],
        'population': ['a', 'c', 'n', 's', 'v', 'y', '?'],
        'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd', '?'],
    }


def get_positive_label():
    return 'e'


def get_negative_label():
    return 'p'


def create_record(csv_record):
    return {
        'label': csv_record[0],
        'cap-shape': csv_record[1],
        'cap-surface': csv_record[2],
        'cap-color': csv_record[3],
        'bruises?': csv_record[4],
        'odor': csv_record[5],
        'gill-attachment': csv_record[6],
        'gill-spacing': csv_record[7],
        'gill-size': csv_record[8],
        'gill-color': csv_record[9],
        'stalk-shape': csv_record[10],
        'salk-root': csv_record[11],
        'stalk-surface-above-ring': csv_record[12],
        'stalk-surface-below-ring': csv_record[13],
        'stalk-color-above-ring': csv_record[14],
        'stalk-color-below-ring': csv_record[15],
        'veil-type': csv_record[16],
        'veil-color': csv_record[17],
        'ring-number': csv_record[18],
        'ring-type': csv_record[19],
        'spore-print-color': csv_record[20],
        'population': csv_record[21],
        'habitat': csv_record[22],
    }


def create_distribution_dict():
    attributes_with_domains = attributes_domains()

    distribution = {}

    for attribute, domains in attributes_with_domains.iteritems():
        if attribute == 'label':
            continue
        for domain in domains:
            distribution[(attribute, domain, 'label', get_positive_label())] = 1
            distribution[(attribute, domain, 'label', get_negative_label())] = 1

    return distribution


def read_file(path=None):
    if path is None:
        path = 'agaricus-lepiota.data'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    records = []
    for value in csv_list:
        records.append(create_record(value))

    return records

# </editor-fold>

# <editor-fold desc="Learn">


def create_distribution_key(attribute, domain_value, label_value):
    return (attribute, domain_value, 'label', label_value)


def put_value_in_distribution(distribution, attribute, domain_value, label_value):
    key = create_distribution_key(attribute, domain_value, label_value)
    distribution[key] += 1


def get_label_count(records, label):
    count = 0
    for record in records:
        if record['label'] == label:
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
    distribution = create_distribution_dict()
    pos_count = get_label_count(records, get_positive_label())
    neg_count = get_label_count(records, get_negative_label())

    for record in records:
        for attribute, domain_value in record.iteritems():
            if attribute == 'label':
                continue
            put_value_in_distribution(distribution, attribute, domain_value, record['label'])

    distribution = create_percentages(pos_count, neg_count, distribution)
    distribution[('label', get_positive_label())] = pos_count / (pos_count + neg_count)
    distribution[('label', get_negative_label())] = neg_count / (pos_count + neg_count)

    return distribution


# </editor-fold>

# <editor-fold desc="Classify">

def calculate_probability_of(distribution, instance, label):
    un_normalized_prob = distribution[('label', label)]
    for attribute, domain_value in instance.iteritems():
        if attribute == 'label':
            continue
        key = create_distribution_key(attribute, domain_value, label)
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
        if record['label'] != classification[0][0]:
            number_of_errors += 1

    return number_of_errors/len(test_data)

# </editor-fold>


# <editor-fold desc="Tests">

test_records = read_file()

random.shuffle(test_records)

half_way = int(math.floor(len(test_records)/2))
set_1 = test_records[:half_way]
set_2 = test_records[half_way:]

distro_1 = learn(set_1)

# Evalutate
c2 = classify(distro_1, set_1)
evaluation_1 = evaluate(set_1, c2)
print "Error Rate = {}".format(evaluation_1)

c2 = classify(distro_1, set_2)
evaluation_1 = evaluate(set_2, c2)
print "Error Rate = {}".format(evaluation_1)




# distribution = learn(data)
# print distribution

distro_2 = learn(set_2)

# Evaluate
c1 = classify(distro_2, set_2)
evaluation_2 = evaluate(set_2, c1)
print "Error Rate = {}".format(evaluation_2)

c1 = classify(distro_2, set_1)
evaluation_2 = evaluate(set_1, c1)
print "Error Rate = {}".format(evaluation_2)



print "new test"

distro_3 = learn(test_records)

# Evaluate
c1 = classify(distro_3, test_records)
evaluation_2 = evaluate(test_records, c1)
print "Error Rate = {}".format(evaluation_2)



# </editor-fold>
