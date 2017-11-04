import numpy as np
import matplotlib.pyplot as plt
import random, copy, math, sys

plain =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,1.0, 1.0, 1.0, 1.0]
forest = [0.0, 1.0, 0.0, 0.0,1.0, 1.0, 1.0, 0.0,1.0, 1.0, 1.0, 1.0,0.0, 1.0, 0.0, 0.0]
hills =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0,0.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0]
swamp =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,1.0, 0.0, 1.0, 0.0,1.0, 1.0, 1.0, 1.0]

figure = plt.figure(figsize=(20,6))

axes = figure.add_subplot(1, 3, 1)
pixels = np.array([255 - p * 255 for p in plain], dtype='uint8')
pixels = pixels.reshape((4, 4))
axes.set_title( "Left Camera")
axes.imshow(pixels, cmap='gray')

axes = figure.add_subplot(1, 3, 2)
pixels = np.array([255 - p * 255 for p in forest], dtype='uint8')
pixels = pixels.reshape((4, 4))
axes.set_title( "Front Camera")
axes.imshow(pixels, cmap='gray')

axes = figure.add_subplot(1, 3, 3)
pixels = np.array([255 - p * 255 for p in hills], dtype='uint8')
pixels = pixels.reshape((4, 4))
axes.set_title( "Right Camera")
axes.imshow(pixels, cmap='gray')

# plt.show()
# plt.close()

#----------------------------------------------------------------------------------------------------------

clean_data = {
    "plains": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]
    ],
    "forest": [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"]
    ],
    "hills": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"]
    ],
    "swamp": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"]        
    ]
}


def view_sensor_image( data):
    figure = plt.figure(figsize=(4,4))
    axes = figure.add_subplot(1, 1, 1)
    pixels = np.array([255 - p * 255 for p in data[:-1]], dtype='uint8')
    pixels = pixels.reshape((4, 4))
    axes.set_title( "Left Camera:" + data[-1])
    axes.imshow(pixels, cmap='gray')
    plt.show()
    plt.close()


# view_sensor_image( clean_data[ "forest"][0])
# view_sensor_image( clean_data["swamp"][0])
#------------------------------------------------------------------------------------------------------------
def blur( data):
    def apply_noise( value):
        if value < 0.5:
            v = random.gauss( 0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss( 0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v
    noisy_readings = [apply_noise( v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]

# view_sensor_image( blur( clean_data["swamp"][0]))
#-----------------------------------------------------------------------------


def re_label(list_of_lists, label):
    for value in list_of_lists:
        value[-1] = label
    return list_of_lists


def relabel_data(data, positive_type_name, positive_alias, negative_alias, ):
    for key in data:
        if key == positive_type_name:
            re_label(data[key], positive_alias)
        else:
            re_label(data[key], negative_alias)

    return data


def get_random_blurred_data(list_of_lists):
    sample_index_to_blur = random.randint(0, len(list_of_lists)-1)
    return blur(list_of_lists[sample_index_to_blur])


def generate_data(data, amount_of_data, positive_type_name):
    data_copy = copy.deepcopy(data)

    positive_alias = 1
    negative_alias = 0

    data_copy = relabel_data(data_copy, positive_type_name, positive_alias, negative_alias)

    positive_data = data_copy[positive_type_name]

    positive_blurred = []
    negative_blurred = []
    x_0 = [1.0]
    for x in xrange(amount_of_data):
        blurred_data = get_random_blurred_data(positive_data)
        feature_vector = x_0 + blurred_data

        positive_blurred.append(feature_vector)

    del data_copy[positive_type_name]

    for x in xrange(amount_of_data):
        random_key_index = random.randint(0, len(data_copy)-1)
        random_key = data_copy.keys()[random_key_index]

        blurred_data = get_random_blurred_data(data_copy[random_key])
        feature_vector = x_0 + blurred_data

        negative_blurred.append(feature_vector)
    
    return positive_blurred + negative_blurred
    

def init_thetas(feature_length):
    thetas = []
    for x in xrange(feature_length): 
        thetas.append(random.uniform(-1, 1))
    return thetas


# def compute_y_hat_comp(thetas, single_feature_set):
#     values = [theta * x_i for theta, x_i in zip(thetas, single_feature_set)]
#     summation = sum(values)
#
#     y_hat = 1/(1 + math.e**(-1*summation))
#     return y_hat


def compute_y_hat(thetas, single_feature_set):
    sum = 0.0
    for j in xrange(len(thetas)):
        sum += thetas[j] * single_feature_set[j]

    y_hat = 1/(1 + math.e**(-1*sum))
    return y_hat


def calculate_error(thetas, data, y_hats):
    count = len(data)

    summation = 0.0
    for index in xrange(len(data)):
        single_feature_set = data[index]
        y_i = single_feature_set[-1]

        # y_hat_i = compute_y_hat(thetas, single_feature_set)
        y_hat_i = y_hats[index]

        partial_sum_1 = 0
        if y_i == 0: 
            partial_sum_1 = 0
        elif y_hat_i == 0:
            partial_sum_1 = -sys.maxint
        else: 
            partial_sum_1 = y_i * math.log(y_hat_i)
        
        partial_sum_2 = 0
        if 1 -y_i == 0: 
            partial_sum_2 = 0
        elif 1 - y_hat_i <= 0:
            partial_sum_2 = -sys.maxint
        else: 
            partial_sum_2 = (1-y_i) * math.log(1-y_hat_i)
        
        summation += partial_sum_1 + partial_sum_2

    error = (-1 * 1/float(count)) * summation

    return error


def logistic_derivative(j, thetas, data, y_hats):
    count = len(data)
    sum = 0.0

    # for single_feature_set in data:
    for data_point in xrange(len(data)):
        single_feature_set = data[data_point]
        y_i = single_feature_set[-1]
        # y_hat_i = compute_y_hat(thetas, single_feature_set)
        y_hat_i = y_hats[data_point]

        inner_value = (y_hat_i - y_i) * single_feature_set[j]

        sum += inner_value

    derivative = (1/float(count)) * sum

    return derivative


def compute_estimates(thetas, data):
    y_hats = []
    for feature_vector in data:
        y_hats.append(compute_y_hat(thetas, feature_vector))
    return y_hats


def gradient_descent(data, alpha=0.1, epsilon=0.0000001, verbose=True):
    ittor_count = 0
    thetas = init_thetas(len(data[0])-1)
    previous_error = 0.0

    y_hats = compute_estimates(thetas, data)
    current_error = calculate_error(thetas, data, y_hats)

    if verbose:
        print current_error

    while abs(current_error - previous_error) > epsilon:
        new_thetas = []
        for j in xrange(len(thetas)):
            new_thetas.append( 
                # thetas[j] - alpha * logistic_derivative(j, thetas, data)
                thetas[j] - alpha * logistic_derivative(j, thetas, data, y_hats)
            )

        thetas = new_thetas

        # previous_error_difference = abs(current_error - previous_error)
        y_hats = compute_estimates(thetas, data)
        previous_error = current_error
        current_error = calculate_error(thetas, data, y_hats)

        if current_error > previous_error:
            print "Adjusting Alpha!!!"
            alpha = alpha / 10

        if verbose and ittor_count % 1000 == 0:
            print "Count: {0} \n Current Error: {1}".format(ittor_count, current_error)

        ittor_count += 1

    return thetas


# Adaptive Alpha
# Return list of thetas
def learn_model(training_data, verbose):
    return gradient_descent(training_data, alpha=.1, verbose=verbose)


# If unlabed, predict
# if labled, return true value, and predicted 
def apply_model(model, test_data, labeled=False):
    threshold = 0.5
    list_of_predictions = []

    for feature_vector in test_data:
        probability = compute_y_hat(model, feature_vector)
        predicted_class = -1
        if probability >= threshold:
            predicted_class = 1
        else:
            predicted_class = 0

        if not labeled:
            list_of_predictions.append((predicted_class, probability))
        else:
            list_of_predictions.append((feature_vector[-1], predicted_class))

    return list_of_predictions


def get_confusion_html(tp, tn, fp, fn, err_rate, tpr, tnr):
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


def calculate_confusion_matrix(results):
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

    html = get_confusion_html(tp, tn, fp, fn, error_rate, true_positive_rate, true_negative_rate)

    return (error_rate, true_positive_rate, true_negative_rate)

####################
#  Test Functions  #
####################

# Generate_data Test
# results = generate_data( clean_data, 10, "hills")
# for result in results:
#     view_sensor_image(result)

# Test learn model!
train_data = generate_data(clean_data, 100, "hills")
model = learn_model(train_data, True)

print model

# model = [-45.78315089096501, -13.800648891179005, -30.387296771892725, -37.41254821759666, -22.32698617835906, 27.94767752962984, 4.627909116402557, 9.657073522270268, 27.65561157860363, -24.079648316348766, 30.11110769084634, 27.923607836505695, -23.041296042836784, -6.759487264562721, 17.75418879675534, 35.28617325323021, -13.410729712858398]

# model = [-8.128004047062685, 2.1450698268535016, -6.380697059707887, 0.026774745706501183, -6.1090844368878265, -5.427263771825765, 0.02942378946926894, -2.2699271308534743, -1.0530132281915288, -1.015112597607468, 5.406389244034168, 9.873463427622521, 1.2729619126389853, -3.074479557830379, 1.3700434488305744, 5.538356595957906, -0.05836080341889052]

test_data = generate_data(clean_data, 100, "hills")

results = apply_model(model, test_data)
print results

results = apply_model(model, test_data, labeled=True)
print results


rates = calculate_confusion_matrix(results)
print rates




#
# test_data = [[1, 1.1, 0], [1, 2.7, 1]]
# thetas = [0.8, 1.1]
# # value = calculate_error(thetas, test_data)
# # print value
#
# derivative = logistic_derivative(0, thetas, test_data)
# print derivative
# theta_0 = thetas[0] - .1 * logistic_derivative(0, thetas, test_data)
# print theta_0
#
#
#
# derivative = logistic_derivative(1, thetas, test_data)
# print derivative
# theta_1 = thetas[1] - .1 * logistic_derivative(1, thetas, test_data)
# print theta_1

