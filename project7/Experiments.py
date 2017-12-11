from Game import Game
from Game import GameState
from QLearning import QLearning

import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def experiment(test_game, num_experiments, file_name, num_episodes=500, alpha=.99, gamma=.9, epsilon=.9, decay_rate=.99):

    list_of_moves_per_experiment = []
    for x in range(num_experiments):
        # Learn model
        q_learning = QLearning(test_game, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, decay_rate=decay_rate)
        q = q_learning.learn()
        num_moves = q_learning.num_moves_per_episode

        list_of_moves_per_experiment.append(num_moves)

    np.array(list_of_moves_per_experiment)
    moves_per_epoc_number = np.sum(list_of_moves_per_experiment, axis=0)
    moves_per_epoc_number = moves_per_epoc_number / num_experiments

    generate_validation_curves(np.arange(num_episodes), moves_per_epoc_number, None, "Number of steps", None,
                               x_axis_label="Epoc Number", y_axis_label="Average Path Length",
                               file_name=file_name)

    return


def generate_validation_curves(x_axis_values, values_line_1, values_line_2, label_1, label_2, x_axis_label="",
                               y_axis_label="Average Epoc Path Length", title="", file_name=""):
    plt.plot(x_axis_values, values_line_1, '-', label=label_1)
    # plt.plot(x_axis_values, values_line_2, '-', label=label_2)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    plt.legend(loc='best')

    if file_name != "":
        plt.savefig(file_name)

    plt.show()
    plt.close()


# <editor-fold desc="Experiments">


# game = Game('tracks/L-track.txt', success_chance=.8)
# experiment(game, 10, "diagrams/L-Track-Easy", num_episodes=2000, alpha=.7, gamma=.7, epsilon=.5, decay_rate=.9)

# game = Game('tracks/R-track.txt', success_chance=.8)
# experiment(game, 10, "diagrams/R-Track-Easy", num_episodes=2000, alpha=.7, gamma=.7, epsilon=.5, decay_rate=.9)


game = Game('tracks/R-track.txt', success_chance=.8, crash_restart=True)
experiment(game, 5, "diagrams/R-Track-Harsh-2", num_episodes=500, alpha=.7, gamma=.7, epsilon=.1)

# x = [3846288, 933924, 20764, 14415, 6180, 9275, 4823, 3421, 25708, 3326, 24999, 5336, 26238, 1275, 7604, 5549, 1068, 17366, 3738, 3243, 25200, 7925, 27143, 38753, 63688, 2543, 13292, 6683, 6432, 71, 9336, 7200, 58864, 13918, 9169, 810, 5813, 58082, 10817, 1726, 613, 2124, 1072, 4726, 1049, 7373, 2799, 1896, 1174, 1738, 1161, 2998, 10158, 831, 432, 1021, 131, 1274, 1899, 842, 107, 2290, 576, 6507, 17373, 10304, 1232, 213, 846, 1919, 2758, 1261, 6822, 4163, 9667, 5118, 2521, 749, 4970, 496, 2971, 684, 803, 1917, 1337, 1316, 7760, 393, 1569, 4133, 5496, 1626, 720, 2482, 11168, 77, 10727, 3663, 2842, 34161, 6518, 22145, 3859, 4997, 4740, 4421, 2447, 5308, 26334, 3574, 7692, 2004, 3964, 4261, 1728, 3219, 1428, 3181, 2355, 723, 943, 2268, 2622, 2079, 607, 1804, 717, 91, 167, 438, 4240, 619, 417, 2433, 1435, 5952, 534, 1412, 4218, 4765, 779, 3848, 9646, 1399, 5308, 4250, 629, 729, 1384, 689, 783, 77, 1026, 1239, 83, 466, 71, 859, 1619, 909, 61, 4330, 462, 374, 15624, 718, 4096, 83, 479, 328, 210, 219, 462, 34, 619, 363, 215, 756, 220, 34, 219, 841, 481, 467, 1032, 110, 582, 421, 149, 1223, 880, 78, 662, 418, 35, 167, 152, 196, 233, 516, 132, 50, 547, 784, 286, 85, 340, 217, 65, 55, 443, 243, 44, 325, 312, 801, 1424, 1793, 296, 4672, 1302, 1964, 3356, 6438, 1170, 3974, 6703, 10459, 2105, 1747, 242, 4111, 2575, 520, 1574, 276, 215, 9590, 6025, 8977, 3138, 3610, 875, 42, 10957, 457, 2938, 1316, 2860, 5812, 6774, 7888, 10481, 8137, 5582, 5140, 572, 5546, 983, 1396, 2622, 2807, 2203, 12829, 1048, 6394, 15006, 8670, 8180, 3558, 3451, 2220, 3793, 1234, 42, 842, 2295, 4809, 742, 1542, 773, 1602, 770, 934, 331, 1787, 34, 1003, 881, 536, 524, 72, 706, 128, 389, 148, 153, 518, 163, 2271, 187, 911, 480, 585, 731, 590, 1313, 1412, 2034, 2479, 1860, 74, 617, 254, 253, 397, 396, 69, 188, 1635, 124, 81, 267, 87, 82, 147, 100, 441, 81, 35, 113, 734, 1393, 75, 276, 291, 402, 364, 107, 238, 730, 36, 54, 231, 174, 432, 440, 31, 35, 252, 742, 30, 467, 1352, 307, 46, 109, 62, 56, 260, 69, 138, 164, 300, 207, 46, 123, 199, 140, 186, 67, 613, 72, 83, 178, 180, 96, 264, 223, 89, 89, 81, 43, 105, 65, 54, 132, 167, 138, 55, 217, 59, 354, 35, 192, 152, 286, 125, 39, 580, 935, 70, 85, 34, 97, 219, 199, 61, 366, 366, 90, 210, 372, 52, 186, 38, 497, 316, 437, 300, 191, 169, 212, 142, 502, 55, 311, 267, 330, 255, 250, 257, 682, 411, 186, 143, 136, 141, 151, 538, 191, 342, 36, 360, 50, 411, 73, 170, 528, 441, 67, 414, 234, 50, 295, 250, 122, 290, 539, 48, 1372, 70, 115, 194, 414, 588, 260, 231, 106, 362, 167, 420, 145, 133, 172, 404, 580, 86, 1127, 321, 388, 233, 171, 105, 41, 67, 144, 48, 244, 209, 51, 851, 111, 138, 203, 682, 35, 363, 272, 248, 115, 239, 80, 65, 657, 395, 398, 180, 138, 143, 121, 82, 296, 234, 337, 169, 331, 313, 105, 97, 188, 42, 122, 199, 138, 105, 818, 197, 120, 100, 828, 251, 87, 80, 35, 748, 847, 97, 135, 214, 74, 100, 98, 139, 134, 986, 194, 106, 132, 53, 188, 101, 289, 68, 140, 74, 221, 38, 63, 341, 35, 114, 193, 58, 43, 261, 287, 78, 237, 92, 162, 88, 234, 151, 162, 102, 338, 194, 500, 139, 173, 154, 488, 350, 227, 94, 134, 209, 88, 66, 369, 308, 183, 77, 533, 485, 140, 70, 129, 195, 524, 98, 375, 874, 38, 82, 141, 127, 417, 38, 571, 240, 102, 246, 78, 722, 176, 723, 63, 86, 222, 314, 211, 360, 1271, 40, 848, 182, 328, 147, 224, 1255, 2778, 572, 143, 40, 60, 128, 184, 143, 38, 79, 676, 44, 378, 61, 470, 188, 36, 35, 285, 188, 70, 125, 166, 40, 286, 212, 107, 310, 81, 68, 68, 54, 126, 88, 214, 529, 827, 244, 44, 140, 752, 110, 345, 424, 127, 407, 172, 187, 132, 197, 32, 40, 46, 88, 59, 57, 247, 279, 50, 56, 470, 285, 50, 187, 35, 95, 85, 40, 174, 48, 100, 39, 51, 556, 176, 46, 299, 302, 65, 120, 108, 272, 54, 51, 266, 388, 178, 58, 73, 62, 77, 144, 52, 69, 35, 88, 161, 55, 114, 88, 51, 62, 64, 491, 134, 62, 33, 114, 34, 38, 112, 152, 44, 39, 87, 36, 31, 60, 86, 226, 79, 36, 51, 45, 49, 264, 41, 38, 240, 80, 170, 33, 120, 187, 33, 118, 80, 227, 68, 102, 58, 35, 246, 38, 349, 356, 495, 238, 78, 31, 335, 87, 35, 57, 320, 156, 155, 67, 56, 33, 74, 34, 146, 32, 94, 83, 272, 195, 349, 75, 61, 66, 88, 67, 59, 32, 43, 32, 803, 190, 73, 668, 43, 296, 184, 126, 353, 80, 251, 467, 75, 99, 172, 61, 79, 228, 238, 78, 36, 328, 282, 130, 48, 235, 35, 34, 148, 143, 32, 153, 121, 59, 305, 82, 178, 414, 114, 175, 162, 481, 164, 206, 267, 86, 30, 62, 745, 375, 773, 197, 85, 49, 107, 135, 101, 222, 162, 57, 31, 42, 33, 462, 33, 308, 36, 53, 48, 123, 36, 33, 37, 62, 142, 29, 43, 150, 39, 52, 92, 39, 54, 161, 36, 78, 184, 67, 63, 78, 76, 204, 31, 96, 65, 199, 34, 201, 49, 51, 91, 43, 117, 103, 113, 57, 251, 113, 54, 150, 45, 39, 152, 110, 49, 156, 39, 147, 98, 32, 94, 93, 34, 122, 78, 55, 89, 221, 184, 39, 69, 101, 33, 87, 129, 32, 260, 63, 706, 84, 254, 119, 56, 229, 152, 95, 40, 44, 169, 76, 42, 34, 40, 83, 94, 167, 38, 169, 175, 55, 146, 202, 277, 57, 316, 157, 148, 30, 65, 103, 566, 172, 43, 197, 284, 138, 127, 89, 53, 79, 62, 198, 159, 116, 237, 42, 136, 115, 234, 70, 37, 94, 66, 108, 51, 233, 112, 53, 50, 61, 116, 121, 211, 175, 40, 109, 79, 52, 33, 128, 43, 184, 179, 45, 320, 166, 38, 89, 34, 95, 92, 210, 62, 53, 71, 31, 76, 84, 32, 146, 50, 93, 234, 112, 37, 189, 33, 36, 94, 109, 236, 124, 93, 177, 78, 118, 74, 52, 139, 102, 67, 346, 207, 91, 152, 287, 74, 195, 97, 108, 65, 35, 193, 64, 36, 167, 39, 110, 176, 82, 44, 352, 236, 40, 263, 413, 280, 109, 383, 38, 248, 143, 381, 71, 137, 39, 156, 117, 288, 33, 45, 110, 282, 39, 108, 60, 89, 503, 103, 53, 41, 75, 563, 116, 219, 112, 130, 279, 165, 85, 357, 151, 223, 35, 88, 194, 100, 275, 127, 665, 110, 220, 112, 284, 78, 217, 35, 54, 61, 97, 452, 424, 91, 219, 70, 337, 68, 91, 230, 80, 63, 153, 88, 92, 123, 72, 260, 342, 69, 461, 160, 233, 126, 194, 173, 357, 178, 110, 899, 54, 165, 59, 253, 231, 63, 189, 38, 85, 77, 103, 61, 370, 40, 171, 153, 49, 87, 34, 400, 38, 134, 162, 341, 120, 268, 208, 162, 71, 327, 40, 41, 41, 35, 54, 62, 104, 221, 84, 586, 518, 413, 370, 102, 70, 113, 35, 43, 71, 194, 154, 439, 59, 118, 224, 48, 70, 152, 105, 51, 135, 152, 186, 112, 152, 296, 79, 48, 122, 823, 92, 174, 35, 73, 422, 530, 140, 81, 157, 248, 274, 60, 81, 46, 515, 246, 76, 265, 65, 172, 299, 91, 36, 230, 147, 39, 347, 210, 60, 313, 63, 103, 216, 40, 67, 51, 303, 86, 68, 90, 189, 154, 486, 310, 102, 220, 48, 43, 187, 111, 205, 438, 533, 106, 53, 69, 465, 703, 61, 84, 127, 43, 37, 44, 109, 134, 75, 175, 259, 187, 65, 112, 61, 72, 76, 36, 165, 277, 62, 102, 76, 35, 60, 525, 116, 89, 59, 51, 141, 52, 142, 115, 158, 44, 162, 86, 276, 60, 151, 60, 86, 40, 32, 177, 83, 49, 304, 47, 59, 160, 292, 47, 35, 98, 249, 266, 68, 163, 260, 306, 146, 53, 40, 117, 64, 207, 74, 47, 127, 205, 60, 58, 149, 144, 61, 265, 113, 85, 214, 147, 236, 90, 85, 138, 61, 278, 52, 182, 43, 31, 133, 181, 155, 143, 99, 128, 139, 466, 441, 40, 87, 37, 55, 40, 344, 227, 278, 50, 39, 511, 150, 102, 97, 413, 64, 197, 81, 66, 38, 53, 179, 85, 358, 351, 62, 33, 44, 61, 303, 585, 102, 260, 74, 202, 34, 147, 35, 290, 60, 81, 410, 159, 314, 138, 99, 94, 117, 91, 265, 63, 99, 58, 111, 236, 439, 156, 119, 315, 618, 153, 142, 92, 62, 52, 32, 247, 239, 161, 63, 140, 78, 37, 122, 187, 52, 255, 344, 129, 222, 39, 92, 202, 41, 96, 392, 912, 143, 377, 725, 41, 365, 123, 503, 90, 500, 371, 38, 51, 279, 98, 378, 157, 66, 120, 33, 124, 62, 60, 100, 118, 79, 397, 92, 93, 111, 119, 281, 51, 33, 154, 194, 512, 46, 100, 86, 329, 305, 37, 128, 186, 218, 255, 57, 646, 88, 42, 162, 140, 52, 73, 195, 59, 346, 44, 79, 67, 198, 323, 209, 219, 116, 118, 249, 79, 172, 220, 327, 74, 40, 191, 409, 135, 121, 40, 65, 56, 256, 275, 123, 141, 116, 95, 91, 48, 275, 170, 75, 59, 189, 37, 291, 61, 159, 70, 156, 172, 267, 248, 130, 80, 125, 544, 63, 73, 67, 99, 35, 217, 33, 184, 119, 38, 122, 139, 48, 217, 41, 47, 49, 199, 49, 201, 557, 135, 123, 36, 264, 50, 239, 256, 305, 123, 40, 593, 247, 294, 78, 169, 112, 124, 96, 420, 297, 149, 71, 57, 175, 212, 353, 48, 658, 454, 125, 179, 41, 37, 163, 53, 147, 207, 274, 164, 48, 124, 60, 392, 48, 70, 121, 133, 174, 45, 153, 153, 88, 227, 235, 242, 118, 264, 65, 35, 326, 73, 220, 349, 37, 90, 227, 934, 105, 100, 210, 226, 138, 85, 204, 159, 36, 64, 159, 501, 227, 35, 88, 36, 43, 263, 90, 75, 218, 38, 38, 179, 36, 249, 37, 351, 256, 312, 281, 666, 222, 342, 88, 103, 109, 98, 77, 100, 63, 54, 126, 179, 109, 164, 38, 185, 74, 207, 346, 109, 218, 271, 196, 40, 37, 108, 100, 54, 537, 192, 106, 37, 50, 68, 39, 78, 125, 63, 172, 63, 52, 73, 39, 143, 138, 77, 38, 74, 239, 36, 67, 53, 99, 116, 35, 130, 42, 46, 77, 36, 37, 338, 60, 35, 66, 116, 271, 82, 35, 42, 161, 101, 37, 70, 62, 146, 256, 269, 38, 54, 89, 54, 88, 44, 147, 37, 101, 124, 38, 84, 116, 88, 43, 106, 120, 62, 60, 103, 105, 88, 214, 36, 216, 51, 42, 200, 42, 51, 93, 78, 170, 94, 34, 49, 306, 37, 38, 159, 113, 221, 437, 123, 274, 74, 108, 54, 64, 120, 203, 42, 78, 35, 37, 68, 124, 59, 42, 165, 80, 41, 154, 40, 38, 43, 86, 100, 48, 129, 113, 168, 44, 300, 35, 46, 96, 166, 109, 181, 66, 67, 34, 54, 50, 36, 80, 77, 37, 38, 286, 35, 78, 114, 62, 70, 45, 178, 39, 53, 128, 35, 61, 91, 92, 37, 51, 101, 168, 74, 61, 67, 72, 149, 64, 36, 106, 52, 45, 37, 49, 65, 70, 38, 162, 141, 70, 79, 59, 78, 46, 77, 58, 38, 36, 100, 46, 40, 59, 66, 298, 68, 40, 351, 99, 43, 60, 293, 36, 121, 66, 96, 39, 262, 45, 53, 94, 54, 143, 129, 85, 37, 37, 183, 39, 47, 355, 55, 48, 206, 258, 37, 76, 101, 89, 94, 50, 52, 143, 97, 147, 38, 245, 93, 81, 271, 40, 65, 38, 120, 48, 58, 52, 36, 39, 74, 160, 74, 64, 54, 73, 44, 154, 112, 209, 77, 74, 746, 239, 144, 68, 325, 44, 204, 53, 143, 161, 52, 905, 93, 347, 50, 205, 219, 666, 92, 164, 181, 42, 89, 123, 112, 121, 47, 43, 105, 85, 188, 41, 82, 44, 177, 271, 85, 42, 249, 40, 33, 40, 73, 181, 45, 270, 228, 72, 89, 74, 106, 185, 80, 37, 114, 155, 36, 36, 60, 41, 216, 109, 175, 106, 37, 104, 489, 68, 80, 664, 50, 220, 331, 64, 83, 92, 337, 317, 94, 45, 86, 45, 39, 358, 207, 157, 286, 192, 281, 457, 56, 44, 66, 599, 544, 189, 243, 238, 101, 150, 238, 71, 38, 37, 88, 55, 664, 49, 66, 40, 97, 54, 45, 39, 181, 78, 70, 82, 41, 93, 37, 147, 176, 161, 42, 152, 123, 40, 150, 42, 236, 143, 184, 401, 196, 36, 263, 58, 86, 456, 216, 72, 43, 152, 116, 144, 629, 157, 112, 359, 76, 122, 95, 450, 101, 38, 514, 75, 84, 149, 307, 211, 617, 209, 42, 66, 145, 397, 141, 44, 160, 263, 134, 138, 74, 54, 100, 39, 50, 38, 82, 317, 203, 50, 338, 107, 116, 299, 171, 170, 79, 432, 360, 299, 84, 64, 270, 146, 172, 133, 49, 90, 199, 45, 183, 544, 299, 294, 82, 71, 41, 148, 115, 163, 41, 67, 57, 539, 69, 43, 46, 350, 40, 118, 181, 138, 50, 39, 81, 89, 121, 127, 163, 313, 381, 137, 40, 40, 290, 31, 81, 197, 61, 42, 50, 78, 204, 78, 480, 250, 41, 346, 295, 720, 222, 75, 88, 44, 41, 82, 129, 234, 155, 310, 174, 45, 118, 172, 87, 43, 193, 200, 88, 203, 516, 188, 64, 95, 85, 40, 257, 205, 124, 43, 88, 76, 501, 205, 69, 204, 50, 222, 41, 155, 712, 133, 131, 102, 104, 135, 308, 135, 126, 79, 379, 116, 146, 110, 400, 332, 45, 41, 428, 186, 224, 233, 130, 50, 132, 368, 37, 103, 39, 226, 50, 390, 346, 68, 135, 201, 106, 70, 406, 222, 227, 102, 226, 89, 150, 121, 308, 123, 160, 182, 198, 167, 347, 364, 39, 297, 58, 760, 219, 551, 678, 66, 661, 50, 372, 201, 76, 83, 159, 167, 427, 76, 45, 281, 243, 117, 153, 211, 86, 290, 48, 184, 137, 320, 173, 71, 94, 46, 106, 37, 503, 279, 212, 213, 196, 38, 82, 862, 403, 905, 181, 356, 447, 382, 87, 701, 273, 1391, 402, 1018, 63, 119, 1134, 136, 48, 355, 273, 41, 38, 295, 172, 36, 39, 38, 122, 135, 40, 274, 52, 72, 83, 129, 35, 575, 38, 96, 121, 43, 399, 46, 218, 40, 40, 132, 949, 71, 66, 220, 92, 200, 341, 56, 133, 160, 90, 405, 347, 127, 198, 408, 70, 744, 72, 205, 43, 166, 135, 586, 85, 82, 81, 51, 270, 143, 352, 298, 663, 519, 573, 70, 638, 85, 80, 52, 269, 99, 88, 105, 271, 100, 157, 255, 57, 70, 90, 482, 163, 324, 114, 97, 156, 230, 59, 381, 51, 130, 399, 127, 82, 147, 1407, 267, 43, 172, 133, 265, 407, 531, 508, 201, 370, 59, 225, 168, 448, 159, 47, 667, 423, 267, 156, 331, 229, 138, 185, 88, 331, 233, 764, 507, 140, 128, 30, 57, 85, 98, 51, 76, 52, 278, 168, 530, 170, 682, 51, 239, 66, 41, 241, 98, 114, 274, 385, 147, 94, 131, 158, 242, 158, 308, 41, 328, 168, 65, 41, 376, 142, 29, 30, 56, 69, 64, 60, 326, 186, 140, 187, 119, 229, 49, 68, 111, 244, 70, 328, 263, 92, 433, 83, 101, 291, 57, 51, 173, 60, 154, 44, 106, 200, 182, 36, 473, 191, 295, 364, 175, 133, 76, 283, 49, 221, 185, 92, 360, 588, 145, 138, 81, 426, 130, 153, 369, 48, 139, 43, 256, 208, 747, 189, 54, 124, 381, 79, 225, 63, 111, 139, 71, 353, 51, 175, 363, 212, 680, 61, 144, 322, 68, 78, 204, 122, 36, 65, 38, 39, 215, 80, 72, 38, 79, 39, 42, 62, 477, 38, 46, 42, 39, 44, 31, 61, 53, 264, 73, 60, 36, 105, 104, 35, 209, 44, 183, 79, 305, 365, 49, 102, 111, 135, 86, 50, 74, 57, 132, 255, 62, 53, 96, 30, 260, 93, 40, 41, 30, 171, 246, 160, 142, 42, 83, 107, 289, 146, 91, 191, 166, 61, 249, 107, 73, 53, 117, 65, 123, 51, 99, 260, 104, 513, 332, 37, 51, 320, 94, 278, 39, 31, 87, 333, 85, 318, 152, 48, 49, 82, 57, 126, 41, 199, 384, 54, 80, 91, 321, 78, 162, 111, 42, 45, 52, 43, 45, 86, 40, 186, 136, 84, 37, 46, 55, 202, 159, 504, 317, 1532, 1307, 331, 621, 105, 85, 755, 1124, 116, 91, 738, 817, 30, 335, 145, 296, 359, 219, 35, 244, 212, 225, 141, 95, 82, 37, 631, 334, 91, 341, 253, 185, 535, 342, 71, 394, 241, 136, 60, 361, 53, 39, 100, 214, 129, 529, 102, 49, 69, 492, 498, 109, 112, 143, 38, 45, 96, 44, 90, 288, 232, 218, 36, 119, 246, 181, 275, 191, 262, 1209, 518, 58, 33, 192, 145, 120, 95, 103, 140, 118, 98, 458, 64, 105, 306, 369, 54, 278, 98, 210, 94, 78, 44, 77, 142, 397, 32, 85, 201, 166, 57, 104, 225, 251, 130, 75, 393, 157, 112, 320, 333, 419, 371, 83, 387, 474, 31, 213, 86, 76, 465, 435, 661, 300, 279, 36, 273, 545, 394, 83, 445, 70, 222, 36, 138, 224, 43, 92, 219, 80, 71, 37, 62, 126, 126, 329, 70, 183, 188, 76, 42, 110, 313, 163, 122, 241, 93, 499, 38, 239, 41, 677, 532, 80, 329, 106, 86, 50, 287, 90, 46, 33, 163, 48, 241, 188, 102, 62, 141, 39, 830, 191, 116, 301, 38, 105, 291, 605, 487, 152, 217, 190, 35, 104, 60, 187, 155, 83, 42, 130, 68, 36, 105, 65, 112, 279, 112, 103, 81, 242, 39, 36, 218, 129, 97, 177, 424, 202, 145, 180, 128, 116, 344, 54, 85, 136, 131, 105, 91, 152, 374, 111, 105, 43, 199, 242, 52, 556, 54, 44, 35, 50, 80, 299, 56, 144, 49, 35, 224, 238, 76, 103, 37, 83, 62, 133, 103, 36, 70, 33, 133, 55, 41, 226, 124, 237, 95, 240, 65, 234, 67, 166, 134, 266, 367, 71, 139, 108, 116, 109, 119, 193, 35, 67, 133, 36, 162, 39, 102, 117, 102, 34, 194, 100, 87, 40, 75, 36, 41, 122, 93, 64, 45, 219, 156, 41, 64, 79, 37, 96, 184, 149, 38, 57, 47, 34, 37, 69, 66, 88, 235, 189, 71, 291, 46, 57, 187, 127, 154, 349, 126, 114, 142, 36, 64, 53, 77, 130, 200, 338, 145, 70, 149, 65, 86, 50, 96, 64, 76, 87, 233, 37, 92, 57, 60, 168, 40, 161, 67, 243, 438, 39, 44, 41, 174, 208, 102, 61, 100, 161, 36, 63, 35, 196, 265, 52, 36, 172, 46, 32, 37, 171, 45, 225, 45, 173, 118, 44, 249, 36, 84, 218, 34, 100, 86, 38, 40, 44, 37, 123, 37, 137, 37, 39, 65, 79, 73, 52, 123, 48, 43, 149, 35, 38, 94, 40, 37, 118, 67, 36, 261, 103, 41, 51, 53, 120, 195, 71, 181, 42, 104, 68, 38, 61, 126, 41, 117, 114, 266, 37, 38, 63, 121, 36, 36, 213, 39, 37, 150, 37, 267, 132, 56, 66, 120, 30, 34, 65, 29, 30, 102, 129, 39, 213, 113, 39, 112, 136, 53, 40, 171, 100, 118, 176, 192, 96, 135, 50, 147, 61, 50, 40, 63, 52, 48, 151, 84, 69, 35, 184, 69, 64, 34, 94, 39, 112, 34, 128, 81, 34, 81, 188, 37, 170, 33, 111, 57, 59, 84, 141, 208, 79, 68, 85, 87, 94, 35, 114, 82, 154, 171, 460, 45, 48, 36, 178, 39, 96, 106, 41, 95, 39, 38, 36, 33, 37, 35, 42, 99, 149, 50, 51, 94, 62, 129, 169, 38, 86, 187, 247, 122, 65, 118, 101, 32, 38, 49, 41, 35, 108, 56, 60, 40, 77, 42, 44, 37, 76, 69, 86, 164, 34, 95, 81, 131, 36, 31, 38, 147, 214, 76, 100, 39, 126, 46, 56, 42, 161, 547, 37, 44, 43, 72, 208, 70, 292, 217, 83, 73, 104, 73, 141, 64, 135, 40, 35, 66, 184, 35, 78, 58, 40, 135, 38, 37, 46, 41, 38, 75, 78, 75, 41, 43, 34, 36, 84, 34, 84, 37, 34, 56, 64, 37, 59, 155, 33, 34, 43, 104, 38, 35, 38, 111, 36, 44, 81, 52, 33, 87, 87, 70, 115, 69, 38, 132, 66, 192, 73, 94, 75, 78, 92, 92, 41, 67, 50, 38, 44, 58, 142, 60, 33, 99, 122, 35, 186, 39, 38, 55, 240, 37, 37, 41, 39, 65, 47, 175, 129, 115, 49, 152, 198, 125, 78, 113, 81, 67, 297, 183, 69, 226, 64, 70, 48, 31, 76, 165, 70, 126, 39, 100, 268, 209, 146, 71, 113, 65, 45, 100, 210, 153, 341, 155, 37, 136, 62, 132, 37, 36, 38, 57, 35, 43, 101, 164, 73, 87, 65, 128, 61, 41, 37, 91, 33, 140, 135, 101, 66, 85, 87, 40, 34, 35, 37, 52, 347, 65, 159, 228, 76, 144, 53, 187, 166, 181, 64, 47, 43, 39, 61, 92, 102, 37, 35, 43, 40, 79, 44, 142, 46, 242, 104, 579, 128, 77, 131, 74, 57, 211, 44, 38, 43, 42, 148, 149, 172, 88, 41, 41, 61, 34, 71, 280, 47, 76, 189, 36, 167, 35, 228, 325, 127, 151, 40, 748, 38, 128, 74, 225, 137, 285, 168, 260, 152, 68, 30, 46, 78, 119, 73, 42, 47, 39, 119, 75, 72, 41, 68, 60, 81, 50, 190, 70, 71, 83, 92, 44, 68, 215, 37, 64, 73, 71, 613, 47, 37, 46, 120, 43, 120, 88, 38, 45, 83, 37, 60, 92, 117, 71, 34, 119, 101, 85, 51, 35, 109, 98, 36, 57, 219, 82, 102, 77, 39, 67, 129, 99, 142, 76, 49, 89, 200, 44, 99, 97, 210, 43, 134, 116, 33, 87, 78, 64, 96, 105, 67, 56, 127, 270, 65, 64, 129, 66, 78, 110, 90, 40, 43, 38, 50, 80, 393, 56, 36, 69, 79, 125, 157, 39, 132, 36, 38, 41, 35, 36, 35, 96, 43, 104, 91, 171, 131, 57, 155, 38, 42, 34, 51, 84, 35, 70, 67, 50, 34, 192, 37, 96, 39, 40, 42, 109, 69, 67, 154, 82, 85, 181, 41, 45, 356, 447, 48, 112, 101, 69, 319, 59, 155, 37, 96, 243, 37, 276, 201, 109, 75, 386, 174, 216, 280, 37, 40, 113, 117, 181, 93, 38, 39, 137, 95, 38, 164, 76, 132, 73, 123, 138, 74, 86, 166, 76, 77, 228, 97, 600, 34, 91, 131, 95, 156, 106, 138, 154, 87, 67, 316, 282, 379, 193, 129, 40, 81, 48, 567, 108, 263, 173, 52, 45, 220, 40, 81, 40, 231, 140, 87, 506, 41, 479, 1294, 167, 163, 991, 282, 217, 705, 160, 44, 468, 152, 610, 72, 258, 203, 135, 30, 170, 307, 42, 85, 58, 454, 84, 127, 35, 179, 73, 153, 32, 111, 266, 236, 60, 92, 74, 571, 39, 126, 33, 362, 166, 751, 187, 670, 229, 66, 616, 38, 176, 205, 389, 129, 35, 50, 434, 62, 64, 107, 238, 143, 150, 91, 69, 501, 54, 140, 78, 353, 32, 68, 67, 38, 39, 32, 128, 54, 40, 46, 36, 64, 72, 36, 92, 38, 113, 65, 38, 59, 178, 204, 73, 115, 32, 79, 34, 42, 149, 132, 223, 82, 79, 126, 161, 106, 161, 118, 33, 101, 75, 39, 183, 34, 122, 120, 49, 256, 59, 296, 41, 97, 106, 96, 49, 32, 85, 28, 45, 32, 98, 37, 38, 183, 103, 433, 129, 140, 47, 103, 39, 151, 80, 132, 113, 278, 123, 110, 67, 35, 124, 76, 71, 49, 132, 156, 160, 197, 44, 33, 31, 39, 76, 98, 130, 75, 92, 209, 76, 34, 35, 86, 34, 185, 36, 65, 57, 103, 80, 77, 177, 127, 73, 172, 34, 79, 32, 67, 149, 72, 106, 35, 60, 29, 37, 53, 55, 39, 72, 59, 42, 36, 73, 35, 63, 62, 135, 100, 82, 111, 38, 89, 158, 41, 38, 37, 32, 187, 34, 80, 125, 179, 38, 244, 135, 107, 37, 105, 36, 37, 35, 149, 35, 56, 72, 147, 43, 148, 41, 43, 42, 82, 37, 147, 115, 82, 63, 34, 45, 35, 58, 90, 62, 35, 37, 175, 91, 94, 39, 66, 81, 37, 127, 39, 35, 42, 35, 46, 108, 76, 51, 148, 34, 71, 42, 30, 130, 41, 132, 36, 209, 69, 34, 35, 51, 57, 82, 46, 101, 55, 164, 36, 38, 49, 84, 65, 65, 74, 69, 39, 38, 86, 35, 39, 146, 39, 37, 49, 72, 72, 42, 87, 148, 130, 37, 39, 37, 35, 35, 41, 39, 57, 109, 34, 35, 44, 78, 81, 63, 152, 40, 43, 36, 76, 61, 38, 147, 47, 115, 39, 52, 121, 148, 78, 49, 36, 65, 73, 238, 120, 133, 35, 81, 85, 37, 39, 203, 57, 67, 39, 170, 94, 45, 36, 37, 82, 71, 52, 41, 120, 40, 90, 169, 136, 40, 121, 39, 40, 64, 34, 39, 91, 38, 89, 149, 128, 98, 78, 80, 43, 98, 38, 40, 37, 33, 103, 36, 35, 30, 38, 68, 80, 111, 41, 48, 75, 50, 39, 126, 78, 111, 43, 38, 90, 32, 131, 40, 130, 37, 56, 38, 35, 39, 116, 40, 263, 93, 68, 39, 179, 55, 38, 36, 67, 125, 31, 39, 39, 161, 187, 78, 94, 147, 55, 93, 136, 96, 48, 56, 227, 247, 148, 145, 143, 202, 203, 101, 135, 36, 72, 43, 90, 47, 40, 71, 34, 66, 1270, 345, 41, 81, 48, 60, 78, 108, 43, 79, 251, 140, 117, 39, 43, 185, 48, 74, 35, 40, 45, 68, 165, 49, 90, 47, 35, 133, 73, 155, 36, 65, 150, 86, 175, 107, 110, 36, 293, 148, 34, 240, 32, 162, 71, 147, 80, 40, 280, 111, 131, 43, 98, 35, 141, 36, 61, 90, 37, 42, 71, 119, 78, 36, 74, 39, 169, 89, 229, 87, 32, 93, 43, 67, 38, 36, 41, 67, 39, 38, 174, 52, 31, 56, 45, 134, 58, 35, 142, 55, 45, 67, 101, 35, 89, 185, 128, 63, 116, 39, 39, 138, 125, 44, 32, 43, 37, 69, 47, 35, 164, 215, 47, 101, 71, 79, 38, 192, 32, 70, 33, 80, 82, 79, 57, 37, 33, 31, 82, 95, 53, 52, 71, 100, 34, 34, 201, 52, 68, 31, 28, 112, 146, 66, 238, 107, 58, 154, 31, 39, 38, 34, 89, 86, 70, 51, 153, 45, 34, 59, 104, 75, 41, 72, 188, 352, 115, 39, 52, 40, 56, 49, 122, 154, 44, 65, 80, 70, 53, 45, 46, 74, 34, 103, 89, 117, 191, 93, 71, 61, 86, 291, 61, 40, 99, 44, 128, 99, 44, 38, 41, 281, 116, 88, 191, 77, 71, 88, 52, 79, 232, 38, 32, 58, 104, 84, 44, 74, 61, 42, 65, 65, 38, 32, 75, 37, 62, 142, 78, 48, 68, 66, 37, 39, 156, 77, 86, 35, 232, 120, 62, 32, 32, 88, 35, 61, 39, 39, 80, 134, 36, 43, 117, 38, 65, 38, 94, 70, 46, 193, 172, 38, 93, 63, 54, 177, 35, 33, 94, 37, 93, 47, 34, 40, 107, 63, 100, 50, 95, 36, 66, 61, 92, 219, 101, 34, 123, 111, 94, 82, 48, 123, 50, 46, 135, 192, 33, 106, 38, 87, 68, 29, 196, 153, 50, 72, 68, 32, 44, 30, 31, 60, 42, 106, 38, 56, 32, 76, 75, 116, 146, 45, 80, 114, 66, 41, 138, 81, 40, 47, 43, 149, 49, 223, 86, 117, 35, 38, 191, 54, 93, 182, 95, 35, 89, 40, 59, 69, 242, 35, 83, 45, 86, 85, 387, 38, 125, 290, 69, 82, 102, 130, 31, 50, 76, 137, 58, 37, 47, 176, 166, 86, 107, 34, 33, 31, 35, 43, 128, 107, 38, 80, 118, 42, 37, 37, 32, 108, 82, 96, 34, 293, 40, 46, 43, 62, 77, 32, 53, 59, 172, 94, 93, 53, 64, 63, 240, 38, 127, 88, 39, 64, 36, 38, 78, 46, 174, 40, 75, 32, 205, 117, 41, 37, 63, 73, 99, 111, 31, 62, 72, 47, 128, 34, 50, 120, 74, 30, 94, 86, 257, 69, 53, 46, 48, 144, 34, 122, 39, 112, 41, 45, 40, 91, 34, 74, 32, 125, 64, 59, 76, 78, 82, 50, 61, 55, 38, 107, 43, 75, 50, 201, 30, 48, 119, 89, 39, 78, 38, 32, 35, 201, 45, 51, 81, 125, 89, 85, 135, 42, 83, 111, 44, 146, 39, 38, 33, 113, 97, 39, 129, 43, 33, 179, 46, 198, 67, 94, 59, 38, 103, 67, 40, 168, 50, 75, 38, 34, 55, 77, 48, 69, 34, 152, 170, 58, 195, 207, 87, 69, 36, 210, 156, 102, 118, 105, 42, 33, 40, 92, 39, 34, 31, 43, 123, 38, 100, 76, 113, 42, 60, 37, 36, 32, 43, 40, 37, 38, 37, 92, 35, 75, 47, 70, 119, 56, 37, 204, 53, 35, 37, 40, 77, 73, 56, 46, 42, 34, 36, 74, 38, 38, 94, 115, 34, 125, 365, 60, 155, 102, 53, 37, 86, 114, 77, 46, 57, 149, 72, 139, 36, 55, 96, 38, 201, 95, 81, 52, 42, 159, 35, 93, 46, 34, 98, 35, 45, 80, 48, 38, 78, 55, 63, 149, 44, 34, 33, 141, 62, 75, 72, 50, 36, 31, 49, 39, 31, 111, 52, 69, 46, 106, 52, 63, 148, 33, 39, 42, 61, 45, 121, 74, 37, 66, 145, 33, 64, 34, 39, 155, 53, 38, 36, 40, 59, 36, 36, 95, 50, 44, 189, 37, 32, 58, 66, 40, 216, 132, 36, 130, 102, 69, 124, 88, 124, 39, 37, 45, 153, 35, 54, 41, 52, 70, 66, 37, 37, 41, 97, 100, 63, 33, 99, 34, 46, 35, 43, 36, 159, 40, 36]
# generate_validation_curves(np.arange(5000), x, None, "Number of steps", None,
#                                x_axis_label="Epoc Number", y_axis_label="Average Path Length",
#                                file_name="diagrams/R-Track-Harsh")



# </editor-fold>

