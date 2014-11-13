"""
Does DBN evaluation on the
ZEMRIS letter dataset.
"""

import util
from dbn import DBN
import numpy as np
import logging

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)
    log.info('DBN evaluation script')

    #   trainset loading
    X, y, classes = util.load_trainset()
    log.info('Read %d samples', len(y))

    #   for testing use only a few classes
    cls_count = len(classes)
    if cls_count < len(classes):
        log.info('Taking a subset of training data')

        cls = ['A', 'B', 'C', 'D', 'E', 'F', 'X', '_BLANK', '_UNKNOWN']
        bool_mask = np.array([(classes[ind] in cls[:cls_count]) for ind in y])
        X = X[bool_mask]
        y = y[bool_mask]
        log.info('Subset has %d elements', len(X))

    test_size = 0.1
    log.info('Splitting data into train and test (%.2f)', test_size)
    test_indices = np.array(
        np.random.binomial(1, test_size, len(X)), dtype=np.bool)
    train_indices = np.logical_not(test_indices)
    X_test = X[test_indices]
    y_test = y[test_indices]
    X_train = X[train_indices]
    y_train = y[train_indices]

    X_mnb, y_mnb = util.create_minibatches(X_train, y_train, 20 * cls_count)

    #   train a DBN
    dbn = DBN([32 * 24, 100, 10], cls_count)
    # lin_eps = util.lin_reducer(0.05, 0.002, 20)
    dbn.train(X_mnb, None, [
        {'epochs': 5, 'eps': 0.05, 'spars': 0.05, 'spars_cost': 6.0},
        {'epochs': 1, 'eps': 0.05}
    ])

    #   visualize first RBM
    util.display_RBM(dbn.rbms[0], 32, 24)

    #   test and evaluate
    y_test_predict = dbn.classify(X_test)
    acc = sum(y_test == y_test_predict) / float(len(y_test))
    f1_score = util.f_macro(y_test, y_test_predict)
    log.info('Prediction accuracy: %.2f, f1-score: %.2f', acc, f1_score)


if __name__ == '__main__':
    main()
