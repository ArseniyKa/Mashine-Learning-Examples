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

        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, filter_size=3, padding=1 )
        self.max_pool1 = MaxPoolingLayer(4,4)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1 )
        self.max_pool2 = MaxPoolingLayer(4,4)
        self.flt = Flattener()
        self.fc = FullyConnectedLayer(2*2, n_output_classes)

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
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        raise Exception("Not implemented!")

    def predict(self, X):
        # You can probably copy the code from previous assignment

        Relu = ReLULayer()
        flt = Flattener()

        Z1 = self.conv1.forward(X)
        relu1 = Relu.forward(Z1)
        pool1 = self.max_pool1.forward(relu1)
        
        Z2 = self.conv2.forward(pool1)
        relu2 = Relu.forward(Z2)
        pool2 = self.max_pool2.forward(relu2)

        flt_arr = self.flt.forward(pool2)
        Z3 = self.fc.forward(flt_arr)

        pred = softmax(Z3)
        pred = np.argmax(pred, axis=1)
        # print("prediction shape is \n", pred.shape)
        assert pred.size == X.shape[0]
        return pred

    def params(self):
        result = {'W1': self.layer1.W, 'B1': self.layer1.B,
                  'W2': self.layer2.W, 'B2': self.layer2.B}
        return result

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        # raise Exception("Not implemented!")

        return result