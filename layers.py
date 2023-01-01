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
    loss = np.sum(loss_arr)
    # loss = np.sum(loss_arr)

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




def l2_regularization(W, B, reg_strength):
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
    B_2 = B*B
    loss = reg_strength * (np.sum(W_2) + np.sum(B_2))
    dW = 2*reg_strength * W
    dB = 2*reg_strength * B
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

    return loss, dW, dB


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        return np.maximum(0, X)

        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_relu = np.greater(self.X, 0).astype(int)
        grad = d_out * d_relu
        # print("grad is \n", grad)
        return grad
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        Z = np.dot(self.X, self.W.value) + self.B.value
        return Z

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        batch_size = d_out.shape[0]
        # print("drelu is \n", drelu)

        self.W.grad = np.dot(self.X.T, d_out)
        ones_arr = np.ones((batch_size, self.B.value.shape[0])).astype(float)
        self.B.grad = np.dot(ones_arr.T, d_out)
        # print("X is\n", self.X)
        # print("B grad is\n", self.B.grad)
        d_input = np.dot(d_out, self.W.value.T)
        # print("dresult is \n", d_result)
        return d_input

        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        raise Exception("Not implemented!")

    def params(self):
        return {'W': self.W, 'B': self.B}
