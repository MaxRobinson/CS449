from __future__ import division
import csv, random, math


def read_file(path=None):
    if path is None:
        path = 'data/ToyExample.txt'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    records = []
    for inner_list in csv_list:
        new_inner_list = []
        for data_point in inner_list:
            if data_point != '?' and inner_list.index(data_point) != len(inner_list)-1:
                new_inner_list.append(int(data_point))
            else:
                new_inner_list.append(data_point)
        records.append(new_inner_list)
        # inner_list[:] = list(int(feature) for feature in inner_list)

    return records


def read_file_float(path=None):
    if path is None:
        path = 'data/ToyExample.txt'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    records = []
    for inner_list in csv_list:
        new_inner_list = []
        for data_point in inner_list:
            if data_point != '?':
                new_inner_list.append(float(data_point))
            else:
                new_inner_list.append(data_point)
        records.append(new_inner_list)
        # inner_list[:] = list(int(feature) for feature in inner_list)

    return records

def read_file_string(path=None):
    if path is None:
        path = 'data/ToyExample.txt'
    with open(path, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    return csv_list


def breast_cancer_munge(data):
    new_data = []

    for record in data:
        new_record = []
        for index in range(len(record)):
            if index == 0:
                continue

            elif index == len(record)-1:
                if record[index] == 2:
                    new_record.append(0)
                else:
                    new_record.append(1)
            else:
                feature = record[index]
                new_feature = [0] * 10

                if feature == '?':
                    feature = random.randint(0, 9)

                new_feature[feature-1] = 1

                for data_point in new_feature:
                    new_record.append(data_point)

        new_data.append(new_record)
    return new_data


def breast_cancer_data():
    data = read_file('data/breast-cancer-wisconsin.data.txt')
    path = 'data/breast-cancer-wisconsin.data.new.txt'

    new_data = breast_cancer_munge(data)

    with open(path, 'w') as f:
        csv_writer = csv.writer(f, lineterminator='\n')

        for record in new_data:
            csv_writer.writerow(record)

def glass_munge(data):
    new_data = []

    for record in data:
        new_record = []
        for index in range(len(record)):
            if index == 0:
                continue
            elif index == len(record)-1:
                new_record.append(record[index])
            else:
                feature = record[index]
                new_feature = [0] * 4

                if feature == '?':
                    feature = random.randint(1, 10)

                new_feature[feature-1] = 1

                for data_point in new_feature:
                    new_record.append(data_point)

        new_data.append(new_record)
    return new_data


def glass_data():
    data = read_file('data/breast-cancer-wisconsin.data.txt')
    path = 'data/breast-cancer-wisconsin.data.new.txt'

    new_data = breast_cancer_munge(data)

    with open(path, 'w') as f:
        csv_writer = csv.writer(f, lineterminator='\n')

        for record in new_data:
            csv_writer.writerow(record)


def soybean_munge(data):
    new_data = []

    for record in data:
        new_record = []
        for index in range(len(record)):
            if index == 0:
                continue
            elif index == len(record)-1:
                new_record.append(record[index])
            else:
                feature = record[index]
                new_feature = [0] * 6

                if feature == '?':
                    feature = random.randint(1, 6)

                new_feature[feature-1] = 1

                for data_point in new_feature:
                    new_record.append(data_point)

        new_data.append(new_record)
    return new_data


def soybean_data():
    data = read_file('data/soybean-small.data.txt')
    path = 'data/soybean-small.data.new.txt'

    new_data = soybean_munge(data)

    with open(path, 'w') as f:
        csv_writer = csv.writer(f, lineterminator='\n')

        for record in new_data:
            csv_writer.writerow(record)



def house_votes_munge(data):
    new_data = []

    for record in data:
        new_record = []
        for index in range(len(record)):
            if index == len(record)-1:
                new_record.append(record[index])
            else:
                feature = record[index]
                new_feature = [0]

                if feature == '?':
                    feature = random.randint(0, 1)
                elif feature == 'y':
                    feature = 1
                else:
                    feature = 0

                new_feature[0] = feature

                for data_point in new_feature:
                    new_record.append(data_point)

        new_data.append(new_record)
    return new_data


def house_votes_data():
    data = read_file_string('data/house-votes-84.data.diff.txt')
    path = 'data/house-votes-84.data.new.txt'

    new_data = house_votes_munge(data)

    with open(path, 'w') as f:
        csv_writer = csv.writer(f, lineterminator='\n')

        for record in new_data:
            csv_writer.writerow(record)


def iris_munge(data):
    new_data = []

    for record in data:
        new_record = []
        for index in range(len(record)):
            feature = record[index]
            new_feature = [0]

            if index == 0:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 7.9

                feature_diff = max_value - feature
                binn = math.floor(feature_diff/.72)

                new_feature[5 - binn] = 1

            elif index == 1:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 4.4

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / .48)

                new_feature[5 - binn] = 1

            elif index == 2:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 6.9

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / 1.18)

                new_feature[5 - binn] = 1
            elif index == 3:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 2.5

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / .54)

                new_feature[5 - binn] = 1

            else:
                new_feature = [record[index]]
                # new_record.append(record[index])
                # continue

            for data_point in new_feature:
                new_record.append(data_point)

        new_data.append(new_record)
    return new_data



def iris_data():
    data = read_file_string('data/iris.data.txt')
    path = 'data/iris.data.new.txt'

    new_data = iris_munge(data)

    with open(path, 'w') as f:
        csv_writer = csv.writer(f, lineterminator='\n')

        for record in new_data:
            csv_writer.writerow(record)
    pass

def glass_munge(data):
    new_data = []

    for record in data:
        new_record = []
        for index in range(len(record)):
            feature = record[index]
            new_feature = [0]

            if index == 0:
                continue

            elif index == 1:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 1.5339

                feature_diff = max_value - feature
                if feature_diff < 0:
                    feature_diff = 0
                binn = math.floor(feature_diff / 0.00454)

                new_feature[5 - binn] = 1

            elif index == 2:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 17.38

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / 1.33)

                new_feature[5 - binn] = 1
            elif index == 3:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 4.49

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / .898)

                new_feature[5 - binn] = 1
            elif index == 4:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 3.5

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / .642)

                new_feature[5 - binn] = 1
            elif index == 5:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 75.41

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / 1.12)

                new_feature[5 - binn] = 1
            elif index == 6:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 6.21

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / 1.242)

                new_feature[5 - binn] = 1
            elif index == 7:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 16.19

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / 2.152)

                new_feature[5 - binn] = 1
            elif index == 8:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = 3.15

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / .63)

                new_feature[5 - binn] = 1

            elif index == 9:
                feature = float(feature)
                new_feature = [0] * 6
                max_value = .51

                feature_diff = max_value - feature
                binn = math.floor(feature_diff / .102)

                new_feature[5 - binn] = 1

            else:
                new_feature = [record[index]]
                # new_record.append(record[index])
                # continue

            for data_point in new_feature:
                new_record.append(data_point)

        new_data.append(new_record)
    return new_data

def glass_data():
    data = read_file_string('data/glass.data.txt')
    path = 'data/glass.data.new.txt'

    new_data = glass_munge(data)

    with open(path, 'w') as f:
        csv_writer = csv.writer(f, lineterminator='\n')

        for record in new_data:
            csv_writer.writerow(record)



# breast_cancer_data()
# soybean_data()
# house_votes_data()
# iris_data()
glass_data()
