"""
Implementation of a deep belief network.
"""

from rbm import RBM
import numpy as np
import logging
import util

log = logging.getLogger(__name__)


class DBN(object):
    """
    Deep Belief Network class.

    A DBN is a stack of RBMs. This Implementation
    provides a way to create the stack and train
    it for a classification task.
    """

    def __init__(self, layer_sizes, class_count):
        """
        Creates a DBN.

        :param layer_sizes: An iterable of integers
            indicating the desired sizes of all the
            layers of the DBN. Note that the fore-last
            layer will be augmented with #class_count
            neurons (which are placed on the start
                of the layer: lowest indices).

        :params class_count: The number of classes
            in the classification tast this DBN is trained for.
        """

        log.info('Creating DBN, classes %d, layer sizes: %r',
                 class_count, layer_sizes)

        self.class_count = class_count

        #   create the RBMs
        rbms = []
        for ind in range(1, len(layer_sizes)):

            #   visible and hidden layer sizes for the RBM
            vis_size = layer_sizes[ind - 1]
            hid_size = layer_sizes[ind]

            #   if making the last RBM, visible layer
            #   get's extra neurons (for class indicators)
            if ind == (len(layer_sizes) - 1):
                vis_size += class_count

            #   create the RBM
            rbms.append(RBM(vis_size, hid_size))

        #   these are the net's RBMs
        self.rbms = rbms

    def classify(self, X):
        """
        Classifies N samples, returns a numpy
        vector of shape (N, ).

        Classification is done with a single
        upward pass through the net, followed
        by a refreshing of the fore-last layer
        of the DBN (the layer that contains class
        indicator neurons). Index of the indicator neuron with
        highest activation probability is the
        classification result.

        :param X: Samples. A numpy array of shape
            (N, n_vis) where n_vis is the number of
            neurons in the lowest (visible) layer of the DBN.
        """

        #   ensure that X is a matrix, not a vector
        if X.ndim == 1:
            X.shape = (X.size, 1)

        #   go through all the RBM's except for the last
        for rbm in self.rbms[:-1]:
            _, X = rbm.hid_given_vis(X)

        #   for the last RBM first append zeros to X
        #   in place where class indicator neurons are
        class_ind = np.zeros((X.shape[0], self.class_count))
        X = np.append(class_ind, X, axis=1)

        #   re-sample the visible layer of the last rbm
        rbm = self.rbms[-1]
        _, X = rbm.hid_given_vis(X)
        X_prb, _ = rbm.vis_given_hid(X)

        #   return index of indicator neuron with
        #   biggest probability
        return np.argmax(X_prb[:, :self.class_count], axis=1)

    def train(self, X_mnb, y_mnb, params):
        """
        Training function for the DBN. Successively
        trains the RBMs of the DBN.

        :param X_mnb: Training data, split into
            minibatches (an iterable of minibatches).
            Each minibatch is a numpy array of shape
            (N, n_vis) where N is the number of samples
            in that minibatch, and n_vis is the number
            of neurons in the visible (lowest) layer
            of the DBN.

        :param y_mnb: Training data labels corresponding to
            X_mnb, split into minibatches (an iterable).
            Each minibatch is a numpy array of shape
            (N, 1).

        :param params: Parameters for training the RBMs.
            An iterable that has (layer_count - 1) elements.
            An element can be 'None', in which case the
            corresponding RBM is not trained (thus allowing
            for RBM training outside the DBN).
            A element is a dict or an iterable that can
            be unpacked into calling arguments to the
            RBM.train(...) method.
        """

        assert len(params) == len(self.rbms)

        #   pre-train one rbm at a time
        rbm_train_res = []
        rbm_X = X_mnb
        for rbm_ind, rbm in enumerate(self.rbms):

            log.info('Training RBM #%d', rbm_ind)

            #   we only train a rbm if we got params for it
            if params[rbm_ind] is None:
                log.info('Skipping RBM #%d training, no params', rbm_ind)
            else:

                #   the last rbm needs labels appended
                if rbm is self.rbms[-1]:

                    log.info('Appending one-hot labes to data')

                    #   iterate through the minibatches
                    for i in range(len(X_mnb)):

                        #   get "one hot" matrix for label indices
                        y_one_hot = util.one_hot(y_mnb[i], self.class_count)

                        #   append one-hot labels in front of the data
                        rbm_X[i] = np.append(y_one_hot, rbm_X[i], axis=1)

                #   train rbm
                if isinstance(params[rbm_ind], dict):
                    rbm_train_res.append(rbm.train(rbm_X, **params[rbm_ind]))
                else:
                    rbm_train_res.append(rbm.train(rbm_X, *params[rbm_ind]))

            #   convert X to input for the next rbm
            rbm_X = [rbm.hid_given_vis(r)[1] for r in rbm_X]

        #   TODO parameter fine-tuning

        return rbm_train_res
