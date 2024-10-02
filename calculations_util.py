from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_fscore_support
from sys import exit
import numpy as np


# for ordinal
trace_label = [1, 0, 0, 0, 0]
debug_label = [1, 1, 0, 0, 0]
info_label = [1, 1, 1, 0, 0]
warn_label = [1, 1 ,1, 1, 0]
error_label = [1, 1, 1, 1, 1]
empty_label = [0, 0, 0, 0, 0] ###adding this label for non calculated label


def toOrdinal(y_list):
    target_list = []

    for y in y_list:
        if y=="trace" or y==0:
            target_list.append(trace_label)
        elif y=="debug" or y==1:
            target_list.append(debug_label)
        elif y=="info" or y==2:
            target_list.append(info_label)
        elif y=="warn" or y==3:
            target_list.append(warn_label)
        elif y=="error" or y==4:
            target_list.append(error_label)
        else:
            target_list.append(empty_label)
    return np.array(target_list)
    
def auc_encoder(y_list):
    target_list = []

    target_trace_label = [1, 0, 0, 0, 0]
    target_debug_label = [0, 1, 0, 0, 0]
    target_info_label = [0, 0, 1 ,0, 0]
    target_warn_label = [0, 0, 0, 1, 0]
    target_error_label = [0, 0, 0, 0, 1]
    target_exception_label = [0, 0, 0, 0, 0]
    for y in y_list:
        if np.array_equal(np.array(y), np.array(trace_label)):
            target_list.append(target_trace_label)
        elif np.array_equal(np.array(y), np.array(debug_label)):
            target_list.append(target_debug_label)
        elif np.array_equal(np.array(y), np.array(info_label)):
            target_list.append(target_info_label)
        elif np.array_equal(np.array(y), np.array(warn_label)):
            target_list.append(target_warn_label)
        elif np.array_equal(np.array(y), np.array(error_label)):
            target_list.append(target_error_label)
        elif np.array_equal(np.array(y), np.array(empty_label)):
            target_list.append(target_exception_label)
        else:
            print("Something wrong happend in auc_encoder.", y)
            target_list.append(target_warn_label)
    return np.array(target_list)


def auc_prob_encoder(y_list):
    target_list = []
    
    target_trace_label = [0.55, 0.20, 0.15, 0.07, 0.03]
    target_debug_label = [0.15, 0.52, 0.30, 0.02, 0.01]
    target_info_label = [0.09, 0.20, 0.60 ,0.09, 0.02]
    target_warn_label = [0.01, 0.03, 0.04, 0.72, 0.20]
    target_error_label = [0.01, 0.02, 0.07, 0.35, 0.55]
    target_exception_label = [0, 0, 0, 0, 0]
    for y in y_list:
        if np.array_equal(np.array(y), np.array(trace_label)):
            target_list.append(target_trace_label)
        elif np.array_equal(np.array(y), np.array(debug_label)):
            target_list.append(target_debug_label)
        elif np.array_equal(np.array(y), np.array(info_label)):
            target_list.append(target_info_label)
        elif np.array_equal(np.array(y), np.array(warn_label)):
            target_list.append(target_warn_label)
        elif np.array_equal(np.array(y), np.array(error_label)):
            target_list.append(target_error_label)
        elif np.array_equal(np.array(y), np.array(empty_label)):
            target_list.append(target_exception_label)
        else:
            print("Something wrong happend in auc_encoder.", y)
            target_list.append(target_info_label)
    return np.array(target_list)


def pd_encoder(y_list): #0:trace, 1:debug, 2:info, 3:warn, 4: error
    target_list = []
    for y in y_list:
        if np.array_equal(np.array(y), np.array(trace_label)):
            target_list.append(0)
        elif np.array_equal(np.array(y), np.array(debug_label)):
            target_list.append(1)
        elif np.array_equal(np.array(y), np.array(info_label)):
            target_list.append(2)
        elif np.array_equal(np.array(y), np.array(error_label)):
            target_list.append(3)
        elif np.array_equal(np.array(y), np.array(warn_label)):
            target_list.append(4)
        elif np.array_equal(np.array(y), np.array(empty_label)):
            target_list.append(3)
        else:
            print("Something wrong happend in pd_encoder.", y)
            target_list.append(3)
    return target_list

def ordinal_accuracy(y_test, y_predicted):

    left_boundary = 0.0
    right_boundary = 4.0
    value_cumulation = 0.0
    for yt, yp in zip(y_test, y_predicted):
        lb_distance = float(yt) - left_boundary
        rb_distance = right_boundary - float(yt)
        max_distance = np.max(np.array([lb_distance, rb_distance]))
        value = 1.0 - abs(float(yp) - float(yt))/max_distance
        value_cumulation = value_cumulation + value
    return value_cumulation/float(len(y_test))


def getMetrics(original_level_as_list, predicted_level_as_list):
    
    labels = toOrdinal(original_level_as_list)
    predictions = toOrdinal(predicted_level_as_list)
    
    auc_y_test = auc_encoder(labels)
    auc_y_predicted = auc_prob_encoder(predictions)
    num_y_test = pd_encoder(labels)
    num_y_predicted = pd_encoder(predictions)
    
    val_auc_ovr_mic = 0
    
    val_oacc = 0
    val_accuracy = accuracy_score(labels, predictions)
    

    

    try:
        val_auc_ovr_mic = roc_auc_score(y_true = auc_y_test, y_score = auc_y_predicted, multi_class="ovr", average = "micro")
    except ValueError:
        pass
    try:
        val_oacc = ordinal_accuracy(num_y_test, num_y_predicted)
    except ValueError:
        pass
    
    return val_accuracy, val_auc_ovr_mic, val_oacc
