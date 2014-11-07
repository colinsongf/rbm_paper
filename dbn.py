"""
Implementation of a deep belief network.
"""

from rbm import RBM
import theano.tensor as T


class DBN(object):

    def __init__(self, layers, classes):

        #   create the RBMs
        rbms = []
        for ind in range(1, len(layers)):
            vis_size = layers[ind - 1]
            hid_size = layers[ind]
            rbms.append(RBM(vis_size, hid_size))

        #   these are the net's RBMs
        self.layers = layers

        #   theano function for calculating the outout
        self.input = T.matrix('input')

    def activations_given_vis(self, input):
        #   TODO implement
        pass

    def train(self, X, y, params):

        assert len(params) == len(self.layers)

        #   TODO train one layer at a time
        layer_X = X
        for layer_ind, layer in enumerate(self.layers):

            #   we only train a layer if we got params for it
            if params[layer_ind] is not None:

                #   TODO last layer needs labels appended
                if layer is self.layers[-1]:
                    pass

                #   TODO train layer

            #   TODO convert X to input for the next layer
            layer_X = layer.hid_given_vis(layer_X)

        #   TODO parameter fine-tuning
