from rbm import RBM
from dbn import DBN
import util
import logging
import numpy as np


log = logging.getLogger(__name__)


def get_data(cls_count=None):
    #   trainset loading
    X, y, classes = util.load_trainset()
    log.info('Read %d samples', len(y))

    #   for testing use only a few classes
    if (cls_count is not None) & (cls_count < len(classes)):
        log.info('Taking a subset of training data')

        cls = ['A', 'B', 'C', 'D', 'E', 'F', 'X', '_BLANK', '_UNKNOWN']
        bool_mask = np.array([(classes[ind] in cls[:cls_count]) for ind in y])
        X = X[bool_mask]
        y = y[bool_mask]
        log.info('Subset has %d elements', len(X))

    return X, y, classes


def test_rbm():

    log.info('Testing RBM')
    rbm = RBM(32 * 24, 300)
    # util.display_RBM(rbm, 32, 24)

    #   trainset loading
    cls_count = 9
    X, y, classes = get_data(cls_count=None)

    #   train the RBM for a while!
    X_mnb = util.create_minibatches(X, None, 20 * cls_count)

    cost, time, hid_act = rbm.train(
        X_mnb, 50, eps=0.05, spars=0.05, spars_cost=0.1)

    util.display_RBM(rbm, 32, 24)


def test_dbn():

    #   trainset loading
    cls_count = 9
    X, y, classes = get_data(cls_count=None)

    X_mnb, y_mnb = util.create_minibatches(X, y, 20 * cls_count)

    lin_eps = util.lin_reducer(0.05, 0.002, 20)
    dbn = DBN([32 * 24, 50, 100], cls_count)
    dbn.train(X_mnb, y_mnb, [
        {'epochs': 20, 'eps': lin_eps, 'spars': 0.125, 'spars_cost': 2.0},
        {'epochs': 20, 'eps': lin_eps}
    ])


def main():
    logging.basicConfig(level=logging.INFO)
    log.info('Test cod for Theano DBN')

    test_rbm()


if __name__ == '__main__':
    main()
