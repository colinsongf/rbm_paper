"""
Multilayer perceptron using Theano (based on Theano tutorial).
"""

import numpy as np
import theano
import theano.tensor as T
from logistic_regression import LogisticRegression
import util
import logging

log = logging.getLogger(__name__)


class HiddenLayer(object):

    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # init W
        if W is None:
            W_values = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            assert W.get_value().shape == (n_in, n_out)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            assert b.get_value().size == n_out

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one or more layers of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, layer_sizes, W=None, b=None):
        """Initialize the parameters for the multilayer perceptron

        :type layer_sizes: iterable of int
        :param layer_sizes: An interable containing layer size counts.

        :type W: iterable of numpy arrays
        :param W: An interable of weights for layer connections. if
            None, new weights are initialized. Note that for a network
            of n layers (1 visible, 1 output and n-1 hidden), W needs
            to have (n - 1) elements, if provided.

        :type b: iterable of numpy arrays
        :param b: An interable of biases for layers. if
            None, new biases are initialized. Note that for a network
            of n layers (1 visible, 1 output and n-1 hidden), W needs
            to have (n - 1) elements, if provided.
        """
        assert((layer_sizes is not None) & (len(layer_sizes) > 1))
        if W is not None:
            assert len(W) == (len(layer_sizes) - 1)
        if b is not None:
            assert len(b) == (len(layer_sizes) - 1)

        self.input = T.matrix('input', dtype=theano.config.floatX)

        #   initialize the hidden layers
        self.hidden_layers = []
        #   first layer is not hidden, last is log. regression
        for layer_ind in range(1, len(layer_sizes) - 1):

            #   figure out the input for layer we're creating
            if layer_ind == 1:
                layer_input = self.input
            else:
                layer_input = self.hidden_layers[-1].output

            #   get layer size, and create it
            n_in = layer_sizes[layer_ind - 1]
            n_out = layer_sizes[layer_ind]

            #   figure out W and b for layer we're creating
            layer_W = None if W is None else W[layer_ind - 1]
            layer_b = None if b is None else b[layer_ind - 1]

            layer = HiddenLayer(layer_input, n_in, n_out, layer_W, layer_b)
            self.hidden_layers.append(layer)

        #   create the logistic regression layer
        reg_input = self.hidden_layers[-1].output
        reg_W = None if W is None else W[-1]
        reg_b = None if b is None else b[-1]
        self.regression_layer = LogisticRegression(
            reg_input, layer_sizes[-2], layer_sizes[-1], reg_W, reg_b)

        #   collect all the params in one place
        self.params = []
        for hid_layer in self.hidden_layers:
            self.params.extend(hid_layer.params)
        self.params.extend(self.regression_layer.params)
        assert len(self.params) == ((len(layer_sizes) - 1) * 2)

    def predict(self, X):
        """
        Predicts labels for given data.
        Returns a numpy array of shape (N, 1)

        :param X: A numpy array of shape (N, n_dim).
        """
        if getattr(self, 'y_pred_f', None) is None:
            self.y_pred_f = theano.function([self.input],
                                            self.regression_layer.y_pred)

        return self.y_pred_f(X)

    def predict_confidence(self, X):
        """
        Predicts label probabilities for given data.
        Returns a numpy array of shape (N, n_lab)
        where n_lab is the number of labels.

        :param X: A numpy array of shape (N, n_dim).
        """
        if getattr(self, 'p_y_given_x_f', None) is None:
            self.p_y_given_x_f = theano.function(
                [self.input], self.regression_layer.p_y_given_x)

        return self.p_y_given_x_f(X)

    def train(self, X_mnb, y_mnb, epochs, eps, weight_cost=1e-4):
        """
        Trains the RBM with the given data. Returns a tuple containing
        (costs, times, hid_unit_activation_histograms). All three
        elements are lists, one item per epoch except for 'times' that
        has an extra element (training start time).

        :param X_mnb: Trainset split into minibatches. Thus,
            X_mnb is an iterable containing numpy arrays of
            (mnb_N, n_vis) shape, where mnb_N is the number of
            samples in the minibatch.

        :param y_mnb: Trainset label indices split into minibatches. Thus,
            y_mnb is an iterable containing numpy arrays of
            (mnb_N, ) shape, where mnb_N is the number of
            samples in the minibatch.

        :type epochs: int
        :param epochs: Number of epochs (int) of training.

        :type eps: float
        :param eps: Learning rate.

        :param weight_cost: Regularization cost for L2 regularization
            (weight decay).
        """

        log.info('Training MLP, epochs: %d, eps: %r',
                 epochs, eps)

        #   inputs to train function are self.input and y for labels
        y = T.ivector('y')
        #   cost function, gradients and updates
        cost = self.regression_layer.negative_log_likelihood(y)
        #   TODO add regularization cost (weights only, or weights and biases)

        return util.cost_minimization(
            [self.input, y], cost, self.params, epochs, eps, X_mnb, y_mnb)
