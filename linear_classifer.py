import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    orig_predictions = predictions.copy()
    print("orig predictions ", orig_predictions)
    dimens = predictions.ndim
    if (dimens == 1):
        max_pred = orig_predictions.max()
        orig_predictions -= max_pred
    else:
        max_pred = orig_predictions.max(axis=1)
        print("max pred is ", max_pred)
        for elem, max_elem in zip(orig_predictions, max_pred):
            elem -= max_elem

    # orig_predictions -= np.max(orig_predictions, axis=dimens - 1)
    # probabilities = np.zeros_like(predictions)
    sum_exps = np.sum(np.exp(orig_predictions),  axis=dimens - 1)
    probabilities = np.exp(orig_predictions)/sum_exps
    print("probabilites are ", probabilities)
    return probabilities
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    # target_arr = np.asarray(target_index, dtype=int)
    # target_arr = target_arr.reshape(1, -1)
    # dimens = target_arr.ndim
    orig_probs = probs.copy()
    orig_probs = orig_probs.reshape(-1)
    loss_arr = np.zeros_like(probs.shape)
    dimens = probs.ndim
    print("targ ind is ", target_index)
    print(target_index.size)
    length = target_index.size
    print("probs ", probs.shape)
    if (length == 1):
        loss = - np.log(orig_probs[target_index])
    else:
        for ls, pr, ind in zip(loss_arr, probs, target_index):
            ls = - np.log(pr[ind])
        loss = np.average(ls)

    # target_arr = probs[target_index]
    # loss = - np.log(probs[target_index])
    print("loss is ", loss)
    return loss

    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value

    '''
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    # print("probs are ", probs)
    # print("loss is ", loss)
    # grad = np.zeros_like(probs.shape)
    orig_probs = probs.copy()
    dimens = 1
    grad = probs.copy()
    print("target index", target_index)
    print(target_index.size)
    length = target_index.size
    if (length == 1):
        old_shape = grad.shape
        grad = grad.reshape(-1)
        grad[target_index] -= 1
        grad = grad.reshape(old_shape)
    else:
        for index in target_index:
            grad[index] -= 1

    # grad = orig_probs
    # grad[target_index] -= 1
    # grad[target_index] = -1/probs[target_index]
    print("grad is ", grad)
    return loss, grad

    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, grad


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

        return y_pred