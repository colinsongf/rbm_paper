from rbm import RBM
import util
import logging
import numpy as np


log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    log.info('Test cod for Theano DBN')

    rbm = RBM(32 * 24, 588)
    # util.display_RBM(rbm, 32, 24)

    #   trainset loading
    X, y, classes = util.load_trainset()
    log.info('Read %d samples', len(y))

    #   for testing use only classes A, B, C
    log.info('Taking a subset of training data')
    classes_mod = 'A'
    bool_mask = np.array([(classes[ind] in classes_mod) for ind in y])
    # X_mod = X[bool_mask]
    X_mod = X
    log.info('Subset has %d elements', len(X_mod))

    #   train the RBM for a while!
    X_mnb = util.create_minibatches(X_mod, None, 20 * len(classes))

    cost, time, hist = rbm.train(
        X_mnb, 10, eps=0.05, spars=0.05, spars_cost=6.0, pcd=True, steps=1)

    util.display_RBM(rbm, 32, 24)


if __name__ == '__main__':
    main()
