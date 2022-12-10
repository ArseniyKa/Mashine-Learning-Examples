import numpy as np


def zero_number_binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    size = prediction.size
    lst = [True if prediction[i] == ground_truth[i]
           else False for i in range(size)]
    accuracy = sum(lst)/size

    # фильтр возьмем те элементы которые на самом деле нули
    zero_arr = prediction[ground_truth]
    print("zero arr is ", zero_arr)
    recall = zero_arr.sum()/zero_arr.size

    predicted_zero_arr = ground_truth[prediction]
    print("predicted_zero_arr  is ", predicted_zero_arr)
    precision = predicted_zero_arr.sum() / predicted_zero_arr.size

    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def nine_number_binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    size = prediction.size
    lst = [True if prediction[i] == ground_truth[i]
           else False for i in range(size)]
    accuracy = sum(lst)/size

    # фильтр возьмем те элементы которые на самом деле нули
    is_nine_number = np.invert(ground_truth)
    nine_arr = prediction[is_nine_number]
    print("nine arr is ", nine_arr)
    recall = len(np.where(nine_arr == False)) / nine_arr.size

    invert_prediction = np.invert(prediction)
    predicted_nine_arr = ground_truth[invert_prediction]
    print("predicted_nine_arr  is ", predicted_nine_arr)
    precision = len(np.where(predicted_nine_arr == False)) / \
        predicted_nine_arr.size

    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
