import numpy as np

from metrics import multiclass_accuracy


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

    dim = orig_predictions.ndim - 1
    max_pred = orig_predictions.max(axis=dim)
    max_pred = max_pred[:, np.newaxis] if dim > 0 else max_pred
    # print("max pred is \n", max_pred)
    orig_predictions -= max_pred
    sum_exps = np.sum(np.exp(orig_predictions),  axis=dim)
    sum_exps = sum_exps[:, np.newaxis] if dim > 0 else sum_exps
    probabilities = np.exp(orig_predictions)/sum_exps

    # print("probabilites are \n", probabilities)
    return probabilities


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
    target_index = target_index.reshape(-1)
    # print("orig_probs \n", orig_probs)
    # print("target index is\n", target_index)
    dim = orig_probs.ndim - 1
    if dim > 0:
        target_probability = orig_probs[range(
            len(orig_probs)), target_index]
    else:
        target_probability = orig_probs[target_index]

    loss_arr = - np.log(target_probability)
    # print("loss array \n", loss_arr)
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
    dim = grad.ndim - 1

    target_index = target_index.reshape(-1)
    # print("target index\n", target_index)
    # первая производная по вероятностям
    if dim > 0:
        grad[range(len(grad)), target_index] -= 1
    else:
        grad[target_index] -= 1

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
    W_2 = W*W
    loss = reg_strength * np.sum(W_2)
    grad = 2*reg_strength * W
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

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
    # вторая производная
    dW = np.dot(X.T, grad)/batch_size

    # # TODO implement prediction and gradient over W
    # # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def initW(self, num_features, num_class):
        self.W = 0.001 * np.random.randn(num_features, num_class)

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
        print("num train is ", num_train)
        print("num features is ", num_features)
        print("num classes  is ", num_classes)

        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        W = self.W
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            # batches_indices =np.array(batches_indices)
            # print("batch indices are\n", (batches_indices.shape))
            for i in range(len(batches_indices)):
                batch_X = X[batches_indices[i]]
                batch_y = y[batches_indices[i]]
                loss, dW = linear_softmax(batch_X, W, batch_y)
                reg_loss, reg_dW = l2_regularization(W, reg)
                W -= learning_rate * (dW + reg_dW)
                loss_history = np.append(loss_history, loss)
                # if(i % 500 == 0):
                # print("Epoch %i, loss: %f" % (epoch, loss))
            test_pred = self.predict(X)
            test_pred = np.argmax(test_pred, axis=1)
            test_accuracy = multiclass_accuracy(test_pred, y)*100
            print('Linear softmax classifier test set accuracy: %f' %
                  (test_accuracy, ))
            print("Epoch %i, loss: %f" % (epoch, loss))

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        predictions = np.dot(X, self.W)
        y_pred = softmax(predictions)
        # y_pred = np.zeros(X.shape[0], dtype=np.int)

        # # TODO Implement class prediction
        # # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")

        return y_pred
