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
    # print("orig predictions \n", orig_predictions)

    max_pred = orig_predictions.max(axis=1)
    # print("max pred is \n", max_pred)
    for elem, max_elem in zip(orig_predictions, max_pred):
        elem -= max_elem
    sum_exps = np.sum(np.exp(orig_predictions),  axis=1)
    # print("sum exps are \n", sum_exps)
    # print("pred shape is ", predictions.shape)
    probabilities = np.zeros(predictions.shape)
    # print("probabilites old \n", probabilities)
    for ind, sums in zip(range(orig_predictions.size), sum_exps):
        probabilities[ind] = np.exp(orig_predictions[ind])/sums

    # print("probabilites are \n", probabilities)
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
    orig_probs = probs.copy()
    orig_probs = orig_probs.reshape(-1)
    loss_arr = np.zeros(probs.shape[0])
    # print("probs ", probs.shape)

    for i, pr, ind in zip(range(loss_arr.size),  probs, target_index):
        # print("probs are \n", pr[ind])
        loss_arr[i] = - np.log(pr[ind])
    print("loss array \n", loss_arr)
    loss = np.average(loss_arr)

    # print("loss is ", loss)
    return loss


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
    orig_probs = probs.copy()
    grad = probs.copy()
    # print("target index\n", target_index)

    for elem, index in zip(grad, target_index):
        # print("index is ", index)
        elem[index] -= 1

    # print("grad is \n", grad)
    return loss, grad


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
    # print("X is \n ", X)
    # print("W is \n ", W)
    predictions = np.dot(X, W)
    loss, grad = softmax_with_cross_entropy(
        predictions, target_index)

    batch_size = target_index.size
    # print("grad is \n", grad)

    batch_dW = np.zeros((batch_size, W.shape[0], W.shape[1]))
    for k in range(batch_size):
        dW_ind = np.zeros(W.shape)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                dW_ind[i][j] = X[k][i] * grad[k][j]

        batch_dW[k] = dW_ind

    # print("batch dW is\n ", batch_dW)
    dW = np.average(batch_dW, axis=0)
    # print("dW is \n", dW)

    # # TODO implement prediction and gradient over W
    # # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

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
