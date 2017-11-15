import numpy as np
import matplotlib.pyplot as plt
import random, copy, math

# <editor-fold desc="Sample">

# plain =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,1.0, 1.0, 1.0, 1.0]
# forest = [0.0, 1.0, 0.0, 0.0,1.0, 1.0, 1.0, 0.0,1.0, 1.0, 1.0, 1.0,0.0, 1.0, 0.0, 0.0]
# hills =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0,0.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0]
# swamp =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,1.0, 0.0, 1.0, 0.0,1.0, 1.0, 1.0, 1.0]
#
# figure = plt.figure(figsize=(20,6))
#
# axes = figure.add_subplot(1, 3, 1)
# pixels = np.array([255 - p * 255 for p in plain], dtype='uint8')
# pixels = pixels.reshape((4, 4))
# axes.set_title( "Left Camera")
# axes.imshow(pixels, cmap='gray')
#
# axes = figure.add_subplot(1, 3, 2)
# pixels = np.array([255 - p * 255 for p in forest], dtype='uint8')
# pixels = pixels.reshape((4, 4))
# axes.set_title( "Front Camera")
# axes.imshow(pixels, cmap='gray')
#
# axes = figure.add_subplot(1, 3, 3)
# pixels = np.array([255 - p * 255 for p in hills], dtype='uint8')
# pixels = pixels.reshape((4, 4))
# axes.set_title( "Right Camera")
# axes.imshow(pixels, cmap='gray')
#
# plt.show()
# plt.close()



# ---------------------------------------------------------------
# </editor-fold>

# <editor-fold desc="Clean Data">

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

# -------------------------------------------------------------------------------
# </editor-fold>

# <editor-fold desc="Blur">
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

# --------------------------------------------------------------------
# </editor-fold>

# <editor-fold desc="Generate Data">

def re_label(list_of_lists, label):
    for value in list_of_lists:
        value[-1] = label
    return list_of_lists


def relabel_data(data):
    _HILL = [1, 0, 0, 0]
    _SWAMP = [0, 1, 0, 0]
    _FOREST = [0, 0, 1, 0]
    _PLAINS = [0, 0, 0, 1]

    for key in data:
        if key == "hills":
            re_label(data[key], _HILL)
        elif key == "swamp":
            re_label(data[key], _SWAMP)
        elif key == "forest":
            re_label(data[key], _FOREST)
        elif key == "plains":
            re_label(data[key], _PLAINS)

    return data


def get_random_blurred_data(list_of_lists):
    sample_index_to_blur = random.randint(0, len(list_of_lists)-1)
    return blur(list_of_lists[sample_index_to_blur])


def generate_data(data, amount_of_data):
    data_copy = copy.deepcopy(data)
    data_copy = relabel_data(data_copy)

    new_data = []

    for key in data_copy:
        for x in range(amount_of_data):
            blurred_data = get_random_blurred_data(data_copy[key])
            new_data.append(blurred_data)

    return new_data

#  </editor-fold>

# <editor-fold desc="Learn_Model">


def generate_neural_net(num_inputs, num_hidden_nodes, num_output_nodes):
    neural_net = {
        'hidden_nodes': [],
        'output_nodes': []
    }

    hidden_nodes_thetas = []
    output_nodes_thetas = []

    for x in range(num_hidden_nodes):
        thetas = [0] * (num_inputs+1)
        hidden_nodes_thetas.append(thetas)

    for x in range(num_output_nodes):
        thetas = [0] * (num_hidden_nodes+1)
        output_nodes_thetas.append(thetas)

    neural_net['hidden_nodes'] = hidden_nodes_thetas
    neural_net['output_nodes'] = output_nodes_thetas

    return neural_net


def init_nn(nn):
    for key in nn:
        for node_index in range(len(nn[key])):
            theta_list = nn[key][node_index]
            nn[key][node_index] = [random.random() for x in range(len(theta_list))]

    return nn


def compute_y_hat(thetas, single_feature_set):
    sum = 0.0
    for j in range(len(thetas)):
        sum += thetas[j] * single_feature_set[j]

    y_hat = 1/(1 + math.e**(-1*sum))
    return y_hat


def calculate_node_output(data_point, nn):
    data_point_w_bias = [1] + data_point

    node_values = {
        'hidden_nodes_output': [],
        'output_nodes_output': []
    }

    hidden_node_thetas = nn['hidden_nodes']
    output_node_thetas = nn['output_nodes']

    for node_theta_list in hidden_node_thetas:
        node_output = compute_y_hat(node_theta_list, data_point_w_bias)
        node_values['hidden_nodes_output'].append(node_output)

    for node_theta_list in output_node_thetas:
        hidden_node_values_w_bias = [1] + node_values['hidden_nodes_output']
        output_node_output = compute_y_hat(node_theta_list, hidden_node_values_w_bias)
        node_values['output_nodes_output'].append(output_node_output)

    return node_values


def calculate_output_node_errors(node_outputs, data_point):
    output_errors = []

    y_output = data_point[-1]
    node_output = node_outputs['output_nodes_output']

    for index in range(len(node_output)):
        y_hat = node_output[index]
        y = y_output[index]
        error = y_hat * (1 - y_hat) * (y - y_hat)
        output_errors.append(error)

    return output_errors


def calculate_theta_times_error(hidden_node_index, output_nodes, output_errors):
    participation_in_error = 0.0
    for output_node_index in range(len(output_nodes)):
        output_node = output_nodes[output_node_index]
        output_error = output_errors[output_node_index]
        corresponding_theta = output_node[hidden_node_index + 1]

        participation_in_error += (corresponding_theta * output_error)

    return participation_in_error


def calculate_hidden_node_errors(nn, node_outputs, output_errors):
    hidden_nodes_outputs = node_outputs['hidden_nodes_output']

    hidden_nodes = nn['hidden_nodes']
    output_nodes = nn['output_nodes']

    hidden_errors = []
    for index in range(len(hidden_nodes)):

        participation_in_output_error = calculate_theta_times_error(index, output_nodes, output_errors)

        y_hat = hidden_nodes_outputs[index]
        hidden_error = y_hat * (1 - y_hat) * participation_in_output_error
        hidden_errors.append(hidden_error)

    return hidden_errors


def update_theta_for_node_list(node_list, input_data, error_list, alpha=.1):
    input_with_bias = [1] + input_data

    for node_index in range(len(node_list)):
        node = node_list[node_index]
        for theta_index in range(len(node)):
            theta = node[theta_index]
            new_theta = theta + alpha * error_list[node_index] * input_with_bias[theta_index]
            node[theta_index] = new_theta

    return node_list


def calculate_total_error_average(nn, data):
    total_errors = [0.0] * len(data[0][-1])

    for data_point in data:
        node_outputs = calculate_node_output(data_point, nn)
        output_errors = calculate_output_node_errors(node_outputs, data_point)

        for index in range(len(total_errors)):
            total_errors[index] = total_errors[index] + output_errors[index]

    for index in range(len(total_errors)):
        total_errors[index] /= float(len(data))

    # mean_summed_error = sum(total_errors)/float(len(total_errors))
    mean_summed_error = sum(total_errors)

    return mean_summed_error
    # return total_errors


def can_stop(current_error, previous_error, epsilon):
    # for c_error, p_error in zip(current_error, previous_error):
    #     if abs(c_error - p_error) > epsilon:
    #         return False
    if abs(current_error - previous_error) > epsilon:
            return False
    return True


def learn_model(data, hidden_nodes, verbose=False):
    nn = generate_neural_net(16, hidden_nodes, 4)
    nn = init_nn(nn)

    epsilon = .0000001

    # previous_error = [0.0] * len(data[-1])
    previous_error = 0.0

    current_error = calculate_total_error_average(nn, data)

    if verbose:
        print(current_error)

    ittor_count = 0
    while not can_stop(current_error, previous_error, epsilon):
        for data_point in data:
            node_outputs = calculate_node_output(data_point, nn)
            output_errors = calculate_output_node_errors(node_outputs, data_point)
            hidden_errors = calculate_hidden_node_errors(nn, node_outputs, output_errors)

            nn['hidden_nodes'] = update_theta_for_node_list(nn['hidden_nodes'], data_point, hidden_errors, alpha=0.1)
            nn['output_nodes'] = update_theta_for_node_list(nn['output_nodes'], node_outputs['hidden_nodes_output'], output_errors, alpha=0.1)

        previous_error = current_error
        current_error = calculate_total_error_average(nn, data)

        if verbose and ittor_count % 1000 == 0:
            print("Count: {0} \n Current Error: {1}".format(ittor_count, current_error))

        ittor_count += 1

    return (nn['hidden_nodes'], nn['output_nodes'])


# </editor-fold>

# <editor-fold desc="Apply_Model">
def get_index_with_max_value(values):
    max_val = -1
    max_val_index = -1

    for index in range(len(values)):
        if values[index] > max_val:
            max_val = values[index]
            max_val_index = index

    return max_val_index


def create_result(outputs, index_with_max, actual_value=[], is_labeled=False):
    result = []

    for index in range(len(outputs)):
        if index == index_with_max:
            if is_labeled:
                result.append((actual_value[index], 1))
            else:
                result.append((1, outputs[index]))
        else:
            if is_labeled:
                result.append((actual_value[index], 0))
            else:
                result.append((0, outputs[index]))

    return result


def apply_model(model, test_data, labeled=False):
    nn = {
        'hidden_nodes': model[0],
        'output_nodes': model[1]
    }

    results = []
    for data_point in test_data:
        outputs = calculate_node_output(data_point, nn)
        final_outputs = outputs['output_nodes_output']
        index_with_max = get_index_with_max_value(outputs['output_nodes_output'])

        if labeled:
            result = create_result(final_outputs, index_with_max, actual_value=data_point[-1], is_labeled=labeled)
        else:
            result = create_result(final_outputs, index_with_max)

        results.append(result)

    return results

# </editor-fold>

# <editor-fold desc="Validation Curves">


def get_predicted_index_for_result(result):
    actual_index = -1
    predicted_index = -1

    for index in range(len(result)):
        if result[index][0] == 1:
            actual_index = index
        if result[index][1] == 1:
            predicted_index = index

    return (actual_index, predicted_index)


def calculate_error_rate(results):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for result in results:
        actual_result = get_predicted_index_for_result(result)
        actual = actual_result[0]
        predict = actual_result[1]

        if actual == predict:
            tp += 1
        elif actual != predict:
            fn += 1
        # elif actual == 0 and actual == predict:
        #     tn += 1
        # elif actual == 0 and actual != predict:
        #     fp += 1

    error_rate = (fn + fp) / float(len(results))
    # true_positive_rate = tp/float((tp+fn))
    # true_negative_rate = tn/float((tn+fp))

    return error_rate


def generate_validation_curves(x_axis_values, values_line_1, values_line_2, label_1, label_2 ):
    plt.plot(x_axis_values, values_line_1, '-', label=label_1)
    plt.plot(x_axis_values, values_line_2, '-', label=label_2)

    plt.legend(loc='best')

    plt.show()
    plt.close()


# </editor-fold>

# <editor-fold desc="helpers">

def ones_hot_to_string(data):
    _HILL = [1, 0, 0, 0]
    _SWAMP = [0, 1, 0, 0]
    _FOREST = [0, 0, 1, 0]
    _PLAINS = [0, 0, 0, 1]

    label = data[-1]
    if label == _HILL:
        return "hills"
    elif label == _SWAMP:
        return "swamp"
    elif label == _FOREST:
        return "forest"
    elif label == _PLAINS:
        return "plains"

    return ""

def ones_hot_to_string_label(label):
    _HILL = [1, 0, 0, 0]
    _SWAMP = [0, 1, 0, 0]
    _FOREST = [0, 0, 1, 0]
    _PLAINS = [0, 0, 0, 1]

    # label = data[-1]
    if label == _HILL:
        return "hills"
    elif label == _SWAMP:
        return "swamp"
    elif label == _FOREST:
        return "forest"
    elif label == _PLAINS:
        return "plains"

    return ""


def show_data(test_data):
    for data in test_data:
        data[-1] = ones_hot_to_string(data)
        view_sensor_image(data)


def show_single_data_point(data_point, label):
    data_point[-1] = " Predicted!: {}".format(ones_hot_to_string_label(label))
    view_sensor_image(data_point)

#  </editor-fold>

# <editor-fold desc="tests">

_HILL = [1, 0, 0, 0]
_SWAMP = [0, 1, 0, 0]
_FOREST = [0, 0, 1, 0]
_PLAINS = [0, 0, 0, 1]

#
# test_data = generate_data(clean_data, 10)
# print len(test_data)
# print len(test_data[0])

# nn = learn_model(None, 2)
# nn = generate_neural_net(2, 3, 2)

#################### SMALL NN TEST ########################
# nn = {}
# nn['hidden_nodes'] = [[.01, .26, -.42], [-.05, .78, .19], [.42, -.23, .37]]
# nn['output_nodes'] = [[.2, .61, .12, -.9], [.3, .28, -.34, .10]]
#
# data_point = [.52, -.97, [1, 0]]
# #
# node_output = calculate_node_output(data_point, nn)
# print node_output
#
# output_error = calculate_output_node_errors(node_output, data_point)
# print output_error
#
# hidden_error = calculate_hidden_node_errors(nn, node_output, output_error)
# print hidden_error
#
#
# nn['hidden_nodes'] = update_theta_for_node_list(nn['hidden_nodes'], data_point, hidden_error, alpha=0.1)
# nn['output_nodes'] = update_theta_for_node_list(nn['output_nodes'], node_output['hidden_nodes_output'], output_error, alpha=0.1)
#
# print nn['hidden_nodes']
# print nn['output_nodes']

#######################################################################


# train_data = generate_data(clean_data, 100)
#
# model = learn_model(train_data, 2, True)
#
# print model

# model = ([[-3.669666837309167, 0.7957505653533951, -2.186162106894712, -0.15833286794515516, -0.8956881927970157, 4.065790204878389, 2.8785270969011805, 3.7104150885535847, 4.276285874410096, 1.6436743810173333, 1.8055110243104557, 2.1508345666981197, 2.1436158267185177, -1.22864978909087, 0.6543697540171407, 0.507009746491148, -2.470931705321849], [2.61412724919053, 1.883575627542445, 1.5864359722238155, 1.5565774996766824, 0.2586648442532777, 1.417721711733239, 3.4816153581567364, 3.9444211987180213, 2.1779693844344648, -2.5671994848096102, -1.251704866886198, -1.7401574541436498, -3.081804060658431, -1.0923723198728437, -1.3286271965902905, -1.530802558667524, 0.10710227493530927]], [[-18.11067492034108, 29.727048963034125, -16.998907049714877], [8.505515568733838, -9.062366270837806, -38.290725275680344], [-14.909568498199686, 2.99777296977161, 17.339307628925074], [0.8542370434190601, -24.514647029049517, 8.36810810423997]])
# model = ([[2.093314245436292, 0.004955618211139324, 0.9233424816053285, 1.5568384152543917, 1.712590484368669, 1.7600841333156463, 4.52815995356409, 3.302997835667522, 1.8091231795928786, -3.4807759327943355, -1.2999083680025327, -0.5012263197260866, -2.660909911530016, -0.005033432714585254, -0.8365679726847912, -0.750878722817293, -1.6699661693155259], [-5.748361110774445, -2.964794075314145, 0.09642744780802318, 0.264895392734491, 2.8612001181785214, 5.899683585049054, 5.02123110452821, 3.1186861599050553, 4.156905215601223, 1.0462269952569312, 2.2354574427695955, 2.6871123354847293, 1.4377061790014123, -0.7789515444110526, -0.7648984441963073, 1.310430142480294, -0.4475624706061164]], [[-22.0644490103448, -17.576374504718427, 34.63772227869235], [9.342900897604816, -44.15487468321881, -6.606706281842025], [-15.07440044677532, 17.344363710610853, 2.785983830954097], [1.0422882469069743, 8.873231566107615, -28.404339722791047]])
# model = ([[1.1095908100577294, 0.12592265665054597, 0.3728186195577938, 0.48210985423949515, 0.1815912274841313, 0.1280595446253784, 0.21721494150238382, 0.7426408917085063, 0.3497866223727105, 0.8299650032538606, 0.017108009544018525, 0.911908480786743, 0.2176480265744214, 0.28163521309467293, 0.6135024591623864, 0.9029181288922441, 0.7579997133737554], [1.0165139072444416, 0.39695256676721263, 0.7535272863776612, 0.3431236313762716, 0.3162746573752485, 0.4288830160239002, 0.25208720244324273, 0.2615931054861628, 0.866563622072375, 0.6642781047408911, 0.03313812470408562, 0.7150669041298292, -0.01574115160257174, 0.9156687816221023, 0.8702465254517859, 0.8335152678798403, 1.0074946774804625], [-0.8769562975066327, 1.501118386180816, 1.0525621556850413, 1.2136377499482534, 1.6056738520865315, 1.1005061061988688, 1.5588864924838945, 1.574416932476464, 1.1378967832900178, 1.0866513079602382, 1.157088906930482, 1.2495058297195873, 0.9599645360052421, -2.1776564731376182, -3.1066626479319024, -3.5581880554400165, -1.6653588474932288], [-0.8386144591447501, -0.0656333812453384, 0.5810619633359451, 0.29100673496409557, 0.12418568742736542, 3.694015647319628, 4.330998601509313, 3.9478962217221536, 3.3954591976060113, 0.4305153941026463, 3.5064213707187295, 3.7673282594464674, 0.5667518379209259, -4.228848568898586, -0.6605501163034689, -0.021907122111293913, -3.9221071218339523], [0.7656083938468163, 0.8583965250747801, 0.4321051447220919, 0.3127545775305731, 0.386359230351263, 0.44601509113398663, 0.6583082317420719, 0.4824600079060915, 0.07724059610954008, 0.7100892439576636, 0.10168797241562705, 0.9529751335110772, 0.5797876243362786, -0.1295861504083911, -0.09272385770602712, 0.5458476328746026, 0.6344928978510634], [0.18610438606618948, 0.3429199774574275, 0.9138047440602177, 0.6015988528618363, 0.9762267274385097, 0.5957864160530677, 0.2776660040854858, 0.8780082610584505, 0.11589816422603298, 0.08852646624958498, 0.787055620245025, 0.08070465011699854, 0.8731146308909505, 0.7961889157332015, 0.4762008234053203, 0.5531428032461951, 0.9107215127801043], [1.1132239582056693, 0.9193341849516785, 0.3473344915858079, 0.20169647355915196, 0.5483047878021922, 0.6507701347581338, 0.6901275174393229, 0.7812245795412971, 0.19339324742973987, 0.11506015334465569, 0.21679953883056033, 0.7761545886383924, 0.7003437552677065, 0.6642875813562843, 1.0240769804422893, 0.7012023578916573, 0.26308666804092834], [-1.9841569034724207, 0.7548400064473454, 0.12784942648644268, 0.4504014988471246, 0.4367349803760095, 0.8223660487480172, 0.13008907657850896, 0.6486465444050591, 0.39461782100863835, 4.504160363677191, 4.676856991828841, 4.831295946526887, 4.592248749066751, -1.5590984049445258, -0.6376879673110566, -0.9768272007777206, -0.8381113755086067], [1.0850921625487804, 0.7035865752811945, 0.46594046806222844, 0.9397947822764487, 0.048496123216864175, 0.08526315611098921, 0.43698926909917507, 0.9301959433763123, 0.252384068969432, 0.45674873064089627, 0.6006869255820497, 0.7195187905323986, 0.7133025981514263, 0.35718695568208464, 0.8288623321807751, 0.6469030929285258, 1.1508550259334072], [0.6371863033960881, 0.15733400563510538, 0.4376468022190337, 0.872356246525245, 0.5750951927150185, 0.1959309822227424, 0.23439347549385137, 0.3624056805426314, 0.7486297024382595, 0.027152272741649882, 0.3420923011685553, 0.7098153094329887, -0.23016242702547904, 0.48568812470585093, 1.1441976354516001, 0.3620679373654971, 0.16635065935445847], [0.9794936229265787, 0.6127803965513932, 0.13299845673086858, 0.40209928482263146, 0.31279910409019634, 0.5210343384839505, 0.5086162475987137, 0.25738050681361796, 0.2120741757576302, 0.3015542164696748, 0.939678280718303, 0.3911898395384252, 0.7255263773695658, 0.6565527765169326, 0.9645588995570502, 0.9723820500751278, 0.32159995953474035], [0.9065379658113093, 0.9198167162323427, 0.44971074990986243, 0.3346439612528774, 0.8212834052824525, 0.7184479953806453, 0.47808401938152406, 0.5142867875319379, 0.8732277578828648, 0.31376078474214475, -0.017489943799714643, 0.35621896355747473, 0.880413196138014, 0.32302551526186113, 0.6763621028086021, 0.4885807744112607, 1.0073150487061249]], [[-0.7177826543598046, -0.08635003535097067, -0.8967173561449374, -8.792237586588234, 9.090878461356297, -0.16667921127187263, -0.7616621045361802, -0.9605942558720243, 1.671514875960731, -0.49129668164191037, -0.6139805036044453, -0.7219052601788216, -0.9851101925629523], [-0.5561129809596456, -0.32922922812461214, -0.7752486051150204, -1.236594528248332, -9.185122715318592, -0.2416176701049051, -0.10827980760209913, -0.3792235814467745, 9.129674344701717, -0.6190598113167903, -1.0327150509226926, -0.1853866255974257, -0.3631769104815045], [-1.0465512346803358, -0.5958430875708347, -0.8418592301708264, 8.803881504641154, 0.748387332210332, -0.11581856348879545, -0.053116004616121516, -0.2511077001749184, 1.0012671865833045, -0.6257127152092938, -0.8450307125053094, -1.1494792976409642, -0.685058027935412], [0.5692996502654402, 0.8018806978206409, 0.8496142622630689, -2.3554178039927334, -1.7458315858267701, -0.253118484495821, 0.4464174474717891, 0.6510753862758283, -8.997103323898363, 0.7371870392107951, 0.07547820898197609, 0.2859344904913429, 0.44616358181017607]])


# test_data = generate_data(clean_data, 300)
#
# results = apply_model(model, test_data, labeled=True)
# results = apply_model(model, test_data)
# print results

# ones_hot_predicition_list = []
# for result in results:
#     ones_hot = []
#     for item in result:
#         ones_hot.append(item[0])
#     ones_hot_predicition_list.append(ones_hot)
#
# for test, prediction in zip(test_data, ones_hot_predicition_list):
#     show_single_data_point(test, prediction)

#
# test_error_rate = calculate_error_rate(results)
# print test_error_rate




########### FINAL TESTS ##########################

# # #
train = generate_data( clean_data, 100)
test  = generate_data( clean_data, 100)


train_error = []
test_error = []

for n in [2, 4, 8]:
    model = learn_model( train, n, verbose=True) # verbose is False now please!

    print("trained: {}".format(n))

    train_results = apply_model( model, train, labeled=True)
    test_results = apply_model( model, test, labeled=True)

    print("applied: {}".format(n))

    train_error_rate = calculate_error_rate(train_results)
    test_error_rate = calculate_error_rate(test_results)

    train_error.append(train_error_rate)
    test_error.append(test_error_rate)


print(train_error)
print(test_error)

generate_validation_curves([2,4,8], train_error, test_error, "Training Data Error", "Test Data Error")


#
# a = [testing_errors['2']['train'], testing_errors['4']['train'], testing_errors['8']['train']]
#
# b = [testing_errors['2']['test'], testing_errors['4']['test'], testing_errors['8']['test']]

# x = np.array([1, 2, 3])
# x_ticks = np.array([2, 4, 8])
#
#
#
# plt.plot([2,4,8], a, '-', label='Training Data')
# plt.plot([2,4,8], b, '-', label='Test Data')
#
# plt.legend(loc='best')
#
# plt.show()


# fig, ax = plt.subplots()
# ax.plot(x, a, 'k--', label='Training Data')
# # ax.plot(x, b, 'k', label='Data length')
#
# ax.xticks(x, x_ticks)
#
# legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# legend.get_frame().set_facecolor('#00FFCC')
# </editor-fold>






























