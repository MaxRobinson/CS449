import numpy as np
import matplotlib.pyplot as plt
import random, copy, math, sys

class LogisticRegression:
    def __init__(self):
        pass

    # def re_label(self, list_of_lists, label):
    #     for value in list_of_lists:
    #         value[-1] = label
    #     return list_of_lists
    #
    # def relabel_data(self, data, positive_type_name, positive_alias, negative_alias, ):
    #     for key in data:
    #         if key == positive_type_name:
    #             self.re_label(data[key], positive_alias)
    #         else:
    #             self.re_label(data[key], negative_alias)
    #
    #     return data

    # <editor-fold desc="Data Pre-process">

    def pre_process(self, data, positive_class_name):
        """
        Pre-processes the data for a given test run.

        The data is preprocessed by taking a positive class label, and modifying the in memory data to replace the
        positive_class_name with a 1, and all other classification names as 0, negative.

        This allows for easier binary classificaiton

        input:
        + data: list of feature vecotrs
        + positive_class_name: Stirng, class to be the positive set.

        """
        new_data = []
        for record in data:
            current_class = record[-1]
            if current_class == positive_class_name:
                record[-1] = 1
            else:
                record[-1] = 0
            new_data.append(record)
        return new_data
        # </editor-fold>

    def init_thetas(self, feature_length):
        thetas = []
        for x in range(feature_length):
            thetas.append(random.uniform(-1, 1))
        return thetas

    def compute_y_hat(self, thetas, single_feature_set):
        sum = 0.0
        for j in range(len(thetas)):
            sum += thetas[j] * single_feature_set[j]

        y_hat = 1/(1 + math.e**(-1*sum))
        return y_hat

    def calculate_error(self, thetas, data, y_hats):
        count = len(data)

        summation = 0.0
        for index in range(len(data)):
            single_feature_set = data[index]
            y_i = single_feature_set[-1]

            # y_hat_i = compute_y_hat(thetas, single_feature_set)
            y_hat_i = y_hats[index]

            partial_sum_1 = 0
            if y_i == 0:
                partial_sum_1 = 0
            elif y_hat_i == 0:
                partial_sum_1 = -sys.maxsize
            else:
                partial_sum_1 = y_i * math.log(y_hat_i)

            partial_sum_2 = 0
            if 1 -y_i == 0:
                partial_sum_2 = 0
            elif 1 - y_hat_i <= 0:
                partial_sum_2 = -sys.maxsize
            else:
                partial_sum_2 = (1-y_i) * math.log(1-y_hat_i)

            summation += partial_sum_1 + partial_sum_2

        error = (-1 * 1/float(count)) * summation

        return error

    def logistic_derivative(self, j, thetas, data, y_hats):
        count = len(data)
        sum = 0.0

        # for single_feature_set in data:
        for data_point in range(len(data)):
            single_feature_set = data[data_point]
            y_i = single_feature_set[-1]
            # y_hat_i = compute_y_hat(thetas, single_feature_set)
            y_hat_i = y_hats[data_point]

            inner_value = (y_hat_i - y_i) * single_feature_set[j]

            sum += inner_value

        derivative = (1/float(count)) * sum

        return derivative

    def compute_estimates(self, thetas, data):
        y_hats = []
        for feature_vector in data:
            y_hats.append(self.compute_y_hat(thetas, feature_vector))
        return y_hats

    def gradient_descent(self, data, alpha=0.1, epsilon=0.0001, verbose=True):
        ittor_count = 0
        thetas = self.init_thetas(len(data[0])-1)
        previous_error = 0.0

        y_hats = self.compute_estimates(thetas, data)
        current_error = self.calculate_error(thetas, data, y_hats)

        if verbose:
            print(current_error)

        while abs(current_error - previous_error) > epsilon:
            new_thetas = []
            for j in range(len(thetas)):
                new_thetas.append(
                    # thetas[j] - alpha * logistic_derivative(j, thetas, data)
                    thetas[j] - alpha * self.logistic_derivative(j, thetas, data, y_hats)
                )

            thetas = new_thetas

            # previous_error_difference = abs(current_error - previous_error)
            y_hats = self.compute_estimates(thetas, data)
            previous_error = current_error
            current_error = self.calculate_error(thetas, data, y_hats)

            if current_error > previous_error:
                print("Adjusting Alpha!!!")
                alpha = alpha / 10

            if verbose and ittor_count % 1000 == 0:
                print("Count: {0} \n Current Error: {1}".format(ittor_count, current_error))

            ittor_count += 1

        return thetas

    # Adaptive Alpha
    # Return list of thetas
    def learn(self, training_data, verbose=False):
        return self.gradient_descent(training_data, alpha=.1, verbose=verbose)

    # If unlabed, predict
    # if labled, return true value, and predicted
    def classify(self, model, test_data, labeled=True):
        threshold = 0.5
        list_of_predictions = []

        for feature_vector in test_data:
            probability = self.compute_y_hat(model, feature_vector)
            predicted_class = -1
            if probability >= threshold:
                predicted_class = 1
            else:
                predicted_class = 0

            if not labeled:
                list_of_predictions.append((predicted_class, probability))
            else:
                list_of_predictions.append((predicted_class, feature_vector[-1]))

        return list_of_predictions

    def get_confusion_html(self, tp, tn, fp, fn, err_rate, tpr, tnr):
        return """
                <table>
                    <tr>
                        <th></th>
                        <th></th>
                        <th colspan="2">Actual</th>
                    </tr>
    
                    <tr>
                        <th></th>
                        <th></th>
                        <th>hill</th>
                        <th>Not hill</th>
                    </tr>
                    <tr>
                        <th rowspan="2">Telephone</th>
                        <td>hill</td>
                        <td>{}</td>
                        <td>{}</td>
                    </tr>
                    <tr>
                        <td>Not hill</td>
                        <td>{}</td>
                        <td>{}</td>
                    </tr>
                </table>
                <p>
                    Error Rate:  {}
                </p>
                <p>
                    True Positive Rate:  {}
                </p>
                <p>
                    True Negative Rate:  {}
                </p>
                """.format(tp, fp, fn, tn, err_rate, tpr, tnr)

    def calculate_confusion_matrix(self, results):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for result in results:
            actual = result[0]
            predict = result[1]

            if actual == 1 and actual == predict:
                tp += 1
            elif actual == 1 and actual != predict:
                fn += 1
            elif actual == 0 and actual == predict:
                tn += 1
            elif actual == 0 and actual != predict:
                fp += 1

        error_rate = (fn + fp) / float(len(results))
        true_positive_rate = tp/float((tp+fn))
        true_negative_rate = tn/float((tn+fp))

        html = self.get_confusion_html(tp, tn, fp, fn, error_rate, true_positive_rate, true_negative_rate)

        return (error_rate, true_positive_rate, true_negative_rate)

####################
#  Test Functions  #
####################

# lr = LogisticRegression()
#
# test_data = [[1, 1.1, 0], [0, 2.7, 1]]
# thetas = [0.8, 1.1]
# # value = calculate_error(thetas, test_data)
# # print( value)
#
# model = lr.learn(test_data, False)
# print(model)
#
# test_point = [[0.2, 2.9, 1]]
# print(lr.classify(model, test_point, True))

# derivative = lr.logistic_derivative(0, thetas, test_data)
# print( derivative)
# theta_0 = thetas[0] - .1 * lr.logistic_derivative(0, thetas, test_data)
# print( theta_0)
#
#
#
# derivative = lr.logistic_derivative(1, thetas, test_data)
# print( derivative)
# theta_1 = thetas[1] - .1 * lr.logistic_derivative(1, thetas, test_data)
# print( theta_1)

