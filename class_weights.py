from config import config
import numpy as np

# mu in "create_class_weight" is a dampening parameter that could be tuned
#
# import numpy as np
# import math
#
#
# def create_class_weight(labels_dict, mu=0.5):
#     total = np.sum(labels_dict.values())
#     keys = labels_dict.keys()
#     class_weight = dict()
#     class_weight_log = dict()
#
#     for key in keys:
#         score = total / float(labels_dict[key])
#         score_log = math.log(mu * total / float(labels_dict[key]))
#         class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
#         class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)
#
#     return class_weight, class_weight_log


# Class abundance for protein dataset
labels_dict = {
    0: 12885,
    1: 1254,
    2: 3621,
    3: 1561,
    4: 1858,
    5: 2513,
    6: 1008,
    7: 2822,
    8: 53,
    9: 45,
    10: 28,
    11: 1093,
    12: 688,
    13: 537,
    14: 1066,
    15: 21,
    16: 530,
    17: 210,
    18: 902,
    19: 1482,
    20: 172,
    21: 3777,
    22: 802,
    23: 2965,
    24: 322,
    25: 8228,
    26: 328,
    27: 11
}
all_labels_dict = {0: 39568,
             1: 3016,
             2: 10510,
             3: 3240,
             4: 5034,
             5: 5777,
             6: 3601,
             7: 9108,
             8: 199,
             9: 189,
             10: 164,
             11: 2133,
             12: 2202,
             13: 1433,
             14: 2630,
             15: 63,
             16: 1270,
             17: 418,
             18: 1843,
             19: 3573,
             20: 426,
             21: 13434,
             22: 2645,
             23: 10138,
             24: 428,
             25: 35783,
             26: 696,
             27: 121}
#
# true_class_weights = create_class_weight(labels_dict)[0]
# log_dampened_class_weights = create_class_weight(labels_dict)[1]

# For smaller dataset
weight_per_class = {0: 3.94, 1: 40.5, 2: 14.02, 3: 32.53, 4: 27.33, 5: 20.21, 6: 50.38, 7: 18.0, 8: 958.15, 9: 1128.49, 10: 1813.64, 11: 46.46, 12: 73.81, 13: 94.57, 14: 47.64, 15: 2418.19, 16: 95.82, 17: 241.82, 18: 56.3, 19: 34.27, 20: 295.24, 21: 13.45, 22: 63.32, 23: 17.13, 24: 157.71, 25: 6.17, 26: 154.82, 27: 4616.55}
log_dampened_class_weights = {0: 1.0, 1: 3.01, 2: 1.95, 3: 2.79, 4: 2.61, 5: 2.31, 6: 3.23, 7: 2.2, 8: 6.17, 9: 6.34, 10: 6.81, 11: 3.15, 12: 3.61, 13: 3.86, 14: 3.17, 15: 7.1, 16: 3.87, 17: 4.8, 18: 3.34, 19: 2.84, 20: 4.99, 21: 1.91, 22: 3.46, 23: 2.15, 24: 4.37, 25: 1.13, 26: 4.35, 27: 7.74}

# For bigger dataset
# weight_per_class = {0: 4.03,
#  1: 52.93,
#  2: 15.19,
#  3: 49.27,
#  4: 31.71,
#  5: 27.63,
#  6: 44.33,
#  7: 17.53,
#  8: 802.22,
#  9: 844.67,
#  10: 973.43,
#  11: 74.84,
#  12: 72.5,
#  13: 111.4,
#  14: 60.7,
#  15: 2534.0,
#  16: 125.7,
#  17: 381.92,
#  18: 86.62,
#  19: 44.68,
#  20: 374.75,
#  21: 11.88,
#  22: 60.36,
#  23: 15.75,
#  24: 373.0,
#  25: 4.46,
#  26: 229.37,
#  27: 1319.36}
# log_dampened_class_weights = {0: 1.0,
#  1: 3.28,
#  2: 2.03,
#  3: 3.2,
#  4: 2.76,
#  5: 2.63,
#  6: 3.1,
#  7: 2.17,
#  8: 5.99,
#  9: 6.05,
#  10: 6.19,
#  11: 3.62,
#  12: 3.59,
#  13: 4.02,
#  14: 3.41,
#  15: 7.14,
#  16: 4.14,
#  17: 5.25,
#  18: 3.77,
#  19: 3.11,
#  20: 5.23,
#  21: 1.78,
#  22: 3.41,
#  23: 2.06,
#  24: 5.23,
#  25: 1.0,
#  26: 4.74,
#  27: 6.49}


weights = np.array(list(log_dampened_class_weights.values()), dtype=float)
batch_weights = weights
for x in range(config.batch_size - 1):
    batch_weights = np.vstack((batch_weights,weights))