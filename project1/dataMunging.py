import csv, random


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

# breast_cancer_data()
soybean_data()
