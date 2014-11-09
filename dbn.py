"""
Implementation of a deep belief network.
"""

from rbm import RBM
import numpy as np
import logging
import util

log = logging.getLogger(__name__)


class DBN(object):

    def __init__(self, layer_sizes, class_count):

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

        assert len(params) == len(self.rbms)

        #   pre-train one rbm at a time
        rbm_X = X_mnb
        for rbm_ind, rbm in enumerate(self.rbms):

            log.info('Training RBM #%d', rbm_ind)

            #   we only train a rbm if we got params for it
            if params[rbm_ind] is not None:

                #   the last rbm needs labels appended
                if rbm is self.rbms[-1]:

                    #   iterate through the minibatches
                    for i in range(len(X_mnb)):

                        #   get "one hot" matrix for label indices
                        y_one_hot = util.one_hot(y_mnb[i], self.class_count)

                        #   append one-hot labels in front of the data
                        rbm_X[i] = np.append(y_one_hot, rbm_X[i], axis=1)

                #   train rbm
                rbm.train_kw(rbm_X, **params[rbm_ind])

            #   convert X to input for the next rbm
            rbm_X = [rbm.hid_given_vis(r)[1] for r in rbm_X]

        #   TODO parameter fine-tuning
