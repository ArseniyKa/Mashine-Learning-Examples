import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    softmax
)


class ConvNet:
    """
    Implements a very simple conv net
    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network
        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """

        self.conv1 = ConvolutionalLayer(
            input_shape[2], conv1_channels, filter_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.max_pool1 = MaxPoolingLayer(2, 2)
        self.conv2 = ConvolutionalLayer(
            conv1_channels, conv2_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.max_pool2 = MaxPoolingLayer(2, 2)
        self.flt = Flattener()
        self.fc = FullyConnectedLayer(8*8*conv2_channels, n_output_classes)

        # TODO Create necessary layers
        # raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        params = self.params()

        params['W1'].grad.fill(0)
        params['B1'].grad.fill(0)
        params['W2'].grad.fill(0)
        params['B2'].grad.fill(0)
        params['W3'].grad.fill(0)
        params['B3'].grad.fill(0)

        Z1 = self.conv1.forward(X)
        assert Z1.shape[1] == X.shape[1]
        assert Z1.shape[2] == X.shape[2]
        # print("Z1 shape ", Z1.shape)
        A1 = self.relu1.forward(Z1)
        pool1 = self.max_pool1.forward(A1)
        # print("pool1 shape ", pool1.shape)

        Z2 = self.conv2.forward(pool1)
        assert Z2.shape[1] == pool1.shape[1]
        assert Z2.shape[2] == pool1.shape[2]
        # print("Z2 shape ", Z2.shape)
        A2 = self.relu2.forward(Z2)
        pool2 = self.max_pool2.forward(A2)
        # print("pool2 shape ", pool2.shape)

        flt_arr = self.flt.forward(pool2)
        # print("flt arr shape ", flt_arr.shape)
        Z3 = self.fc.forward(flt_arr)
        # print("Z3 shape ", Z3.shape)
        loss, grad = softmax_with_cross_entropy(Z3, y)
        grad/=X.shape[0]

        d_input3 = self.fc.backward(grad)
        d_input3 = self.flt.backward(d_input3)

        d_pool2 = self.max_pool2.backward(d_input3)
        d_relu2 = self.relu2.backward(d_pool2)
        d_input2 = self.conv2.backward(d_relu2)
        # print("d_input2 \n ", d_input2)


        d_pool1 = self.max_pool1.backward(d_input2)
        # print("d_pool1 \n ", d_pool1)
        d_relu1 = self.relu1.backward(d_pool1)
        # print("d_relu1 \n ", d_relu1)
        d_input1 = self.conv1.backward(d_relu1)

        return loss

        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        # raise Exception("Not implemented!")

    def predict(self, X):
        # You can probably copy the code from previous assignment

        Z1 = self.conv1.forward(X)
        assert Z1.shape[1] == X.shape[1]
        assert Z1.shape[2] == X.shape[2]
        # print("Z1 shape ", Z1.shape)
        A1 = self.relu1.forward(Z1)
        pool1 = self.max_pool1.forward(A1)
        # print("pool1 shape ", pool1.shape)

        Z2 = self.conv2.forward(pool1)
        assert Z2.shape[1] == pool1.shape[1]
        assert Z2.shape[2] == pool1.shape[2]
        # print("Z2 shape ", Z2.shape)
        A2 = self.relu2.forward(Z2)
        pool2 = self.max_pool2.forward(A2)
        # print("pool2 shape ", pool2.shape)

        flt_arr = self.flt.forward(pool2)
        # print("flt arr shape ", flt_arr.shape)
        Z3 = self.fc.forward(flt_arr)
        # print("Z3 shape ", Z3.shape)

        pred = softmax(Z3)
        pred = np.argmax(pred, axis=1)
        # print("prediction shape is \n", pred.shape)
        assert pred.size == X.shape[0]
        return pred

    def params(self):
        result = {'W1': self.conv1.W, 'B1': self.conv1.B,
                  'W2': self.conv2.W, 'B2': self.conv2.B,
                  'W3': self.fc.W, 'B3': self.fc.B}
        return result

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        # raise Exception("Not implemented!")

        return result
