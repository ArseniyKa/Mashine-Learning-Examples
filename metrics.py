import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    coinsence_count = 0
    size = prediction.size
    for i in range(size):
        if prediction[i] == ground_truth[i]:
            coinsence_count += 1

    return coinsence_count/size
    # # TODO: Implement computing accuracy
    # return 0
