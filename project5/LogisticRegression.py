import random, copy, math, sys

class LogisticRegression:
    def __init__(self):
        pass

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
        """
        Helper function to initialize the weights, thetas
        :param feature_length: length of the feature vector
        :return: list of thetas
        """
        thetas = []
        for x in range(feature_length):
            thetas.append(random.uniform(-1, 1))
        return thetas

    def compute_y_hat(self, thetas, single_feature_set):
        """
        The function that computes the estimated value of a given data point.
        Uses 1/ (1+ e^sum(w_i + x_i))
        :param thetas:
        :param single_feature_set:
        :return:
        """
        sum = 0.0
        for j in range(len(thetas)):
            sum += thetas[j] * single_feature_set[j]

        y_hat = 1/(1 + math.e**(-1*sum))
        return y_hat

    def calculate_error(self, thetas, data, y_hats):
        """
        Calculates the error between the predicted values, y_hats and the data.
        This does so in a batched manner.

        The error rate is calculated assuming 2 possible classes, 0 and 1.
        error =  (y)* log(y_hat) + (1 - y)* log(y_hat)

        :param thetas: list of thetas  the model
        :param data: list of list of datapoints
        :param y_hats: list of predictions for each datapoint
        :return: error rate
        """
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
        """
        Calculates the derivative of the objective function for logistic regression.
        This is the function that provides an update the models.

        The result is used to move along the gradient for gradient descent.

        does this in a batched manner. Looks at all predictions at once.
        used to update a single theta at a time.

        :param j: jth weight to update.
        :param thetas: list of thetas
        :param data: list of data points
        :param y_hats: list of predicted values.
        :return: the amount to move down the gradient.
        """
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
        """
        Computes all estimated values for the data given a model.

        :param thetas: model
        :param data: dataset
        :return: list of predictions.
        """
        y_hats = []
        for feature_vector in data:
            y_hats.append(self.compute_y_hat(thetas, feature_vector))
        return y_hats

    def gradient_descent(self, data, alpha=0.1, epsilon=0.0001, verbose=True):
        """
        The main workhorse for learning the logistic model.

        Uses gradient descent with a convergence factor of epsilon.
        The size of the steps taken down the gradient are given by alpha ( a proportion of the step to take)


        :param data: list of datapoints
        :param alpha: step size
        :param epsilon: convergence value.
        :param verbose: if to print debug
        :return: a list of floats, the learned model
        """
        ittor_count = 0
        thetas = self.init_thetas(len(data[0])-1)
        previous_error = 0.0

        y_hats = self.compute_estimates(thetas, data)
        current_error = self.calculate_error(thetas, data, y_hats)

        if verbose:
            print(current_error)

        # Start main loop
        # if error is less then epsilon stop.
        while abs(current_error - previous_error) > epsilon:
            # Calculate the new thetas
            new_thetas = []
            for j in range(len(thetas)):
                new_thetas.append(
                    thetas[j] - alpha * self.logistic_derivative(j, thetas, data, y_hats)
                )

            thetas = new_thetas

            # Get the estimated values for all of the data points
            y_hats = self.compute_estimates(thetas, data)
            # calculate the error
            previous_error = current_error
            current_error = self.calculate_error(thetas, data, y_hats)

            # Adjust alpha if it looks like we are taking to big of steps.
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
        """
        Entry function to start gradient descent.

        :param training_data: list of data points
        :param verbose: to print or not.
        :return: Model
        """
        return self.gradient_descent(training_data, alpha=.1, verbose=verbose)

    # If unlabed, predict
    # if labled, return true value, and predicted
    def classify(self, model, test_data, labeled=True):
        """
        Function to classify data.
        Calculates the estimated probabilities for each data point based on the model.
        The probability and the predicted class are then added to the list of predictions

        :param model:  list of thetas
        :param test_data: test data
        :param labeled: if the data is labeled
        :return: list of tuples of predictions ( prediction , actual class)
        """
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

