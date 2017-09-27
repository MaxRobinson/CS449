""" Created by Max 9/23/2017 """
import csv


class CustomCSVReader:

    @staticmethod
    def read_file(path=None, data_type=float):
        """
        Helper function to read in CSV data and parse the correct type.
        :param path: Path to file to read
        :param data_type: type of the data in the file
        :return: list of list of data points from csv file.
        """
        if path is None:
            path = 'data/breast-cancer-wisconsin.data.txt'
        with open(path, 'r') as f:
            reader = csv.reader(f)
            csv_list = list(reader)

        for inner_list in csv_list:
            new_values = []
            for feature in inner_list:
                try:
                    new_feature = data_type(feature)
                    new_values.append(new_feature)
                except Exception:
                    new_values.append(feature)

            inner_list[:] = new_values

        return csv_list