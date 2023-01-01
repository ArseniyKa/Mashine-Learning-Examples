import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.reg = reg
        # TODO Create necessary layers
        # raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # self.params()
        Relu = ReLULayer()

        self.layer1.W.grad = np.zeros_like(self.layer1.W.value)
        self.layer1.B.grad = np.zeros_like(self.layer1.B.value)

        self.layer2.W.grad = np.zeros_like(self.layer2.W.value)
        self.layer2.B.grad = np.zeros_like(self.layer2.B.value)

        Z1 = self.layer1.forward(X)
        relu = Relu.forward(Z1)
        Z2 = self.layer2.forward(relu)
        loss, grad = softmax_with_cross_entropy(Z2, y)
        d_input2 = self.layer2.backward(grad)
        drelu = Relu.backward(d_input2)
        d_input1 = self.layer1.backward(drelu)

        reg_loss1, reg_dW1, reg_dB1 = l2_regularization(
            self.layer1.W.value, self.layer1.B.value, self.reg)
        reg_loss2, reg_dW2, reg_dB2 = l2_regularization(
            self.layer2.W.value, self.layer2.B.value, self.reg)
        self.layer1.W.grad += reg_dW1
        self.layer2.W.grad += reg_dW2

        self.layer1.B.grad += reg_dB1
        self.layer2.B.grad += reg_dB2

        loss += reg_loss1 + reg_loss2
        return loss

        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        raise Exception("Not implemented!")

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        Relu = ReLULayer()
        Z1 = self.layer1.forward(X)
        relu = Relu.forward(Z1)
        Z2 = self.layer2.forward(relu)
        pred = softmax(Z2)
        pred = np.argmax(pred, axis=1)
        # print("prediction shape is \n", pred.shape)
        assert pred.size == X.shape[0]
        return pred

        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        # pred = np.zeros(X.shape[0], np.int)

        # raise Exception("Not implemented!")

    def params(self):

        result = {'W1': self.layer1.W, 'B1': self.layer1.B,
                  'W2': self.layer2.W, 'B2': self.layer2.B}
        return result
        # TODO Implement aggregating all of the params

        raise Exception("Not implemented!")

        return result
