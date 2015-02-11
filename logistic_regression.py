"""
Modified Theano tutorial that introduces logistic regression using Theano
and stochastic gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}

The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)

This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

import numpy
import theano
import theano.tensor as T
import numpy as np
import logging
import util

log = logging.getLogger(__name__)


class LogisticRegression(object):

    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch). Typically a symbolic
                      varlible that at runtime resolves to a (N, n_in)
                      matrix, where N is the number of samples in batch.

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type W: TheanoShared
        :W: Weights parameter of the model. Theano shared variable
            backed by a (n_in, n_out) matrix. If None, then it is created
            using random initialization.

        :type b: TheanoShared
        :b: Bias parameter of the model. Theano shared variable
            backed by a (1, n_out) matrix. If None, then it is created
            using random initialization.

        """

        self.input = input

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            b = theano.shared(
                value=numpy.zeros(
                    n_out,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        self.b = b

        # symbolic expression for the matrix of class-membership probs
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description for classification
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def predict(self, X):
        """
        Predicts labels for given data.
        Returns a numpy array of shape (N, 1)

        :param X: A numpy array of shape (N, n_dim).
        """
        if getattr(self, 'y_pred_f', None) is None:
            self.y_pred_f = theano.function([self.input], self.y_pred)

        return self.y_pred_f(X)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label index. Resolves to a matrix of shape
                  (N, n_out)

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

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

        log.info('Training Logistic regression, epochs: %d, eps: %r',
                 epochs, eps)

        #   inputs to train function are self.input and y for labels
        y = T.ivector('y')
        #   cost function, gradients and updates
        cost = self.negative_log_likelihood(y) \
            + weight_cost * T.sum(self.W ** 2)

        return util.cost_minimization(
            [self.input, y], cost, self.params, epochs, eps, X_mnb, y_mnb)


def main():
    logging.basicConfig(level=logging.INFO)
    log.info("Testing logistic regression class")

    #   generate some data
    #   centers, per class, per dimension
    centers = [[1, 1, 1], [1, -1, 1], [-1, 1, -1]]
    cls_count = len(centers)
    n_dim = len(centers[0])

    #   variances, per class, per dimenzion
    vars = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert (len(vars) == cls_count)

    N_per_class = 2500
    N = N_per_class * cls_count
    log.info("Generating data, %d classes, %d samples per class",
             cls_count, N_per_class)
    X = np.zeros((N, n_dim))
    y = np.zeros(N, dtype=np.int32)
    for i in range(N):
        cls = i / N_per_class
        y[i] = cls
        for dim in range(n_dim):
            X[i, dim] = np.random.normal(centers[cls][dim], vars[cls][dim])

    log.info("Splitting into train and test sets")
    train_mask = np.random.rand(N) < 0.85
    test_mask = np.logical_not(train_mask)
    X_train = X[train_mask]
    y_train = y[train_mask]
    log.info("%d samples in train set", len(X_train))

    log.info("Creating minibatches")
    X_mnb, y_mnb = util.create_minibatches(X_train, y_train, cls_count * 10)

    log.info("Fitting")
    estimator = LogisticRegression(T.matrix("input"), n_dim, cls_count)
    log.info("Init acc: %.2f", util.acc(
        y[test_mask], estimator.predict(X[test_mask])))
    for i in range(10):
        estimator.train(X_mnb, y_mnb, 1, 0.1)
        #   validate
        acc = util.acc(y[test_mask], estimator.predict(X[test_mask]))
        log.info("Current acc: %.2f", acc)


if __name__ == '__main__':
    main()
