from rbm import RBM
from dbn import DBN
import util
import logging
import numpy as np
import workflow as wf
import analysis


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
    rbm = RBM(32 * 24, 100)
    analysis.display_RBM(rbm, 32, 24)

    #   trainset loading
    cls_count = 9
    X, y, classes = get_data(cls_count=cls_count)

    #   train the RBM for a while!
    X_mnb = util.create_minibatches(X, None, 20 * cls_count)

    cost, time, hid_act = rbm.train(
        X_mnb, **{'epochs': 5, 'eps': 0.05, 'spars': 0.05, 'spars_cost': 6.0})

    analysis.display_RBM(rbm, 32, 24)


def test_dbn():

    log.info('Testing DBN')

    #   trainset loading
    cls_count = 9
    X, y, classes = get_data(cls_count=None)

    X_mnb, y_mnb = util.create_minibatches(X, y, 20 * cls_count)

    # lin_eps = util.lin_reducer(0.05, 0.002, 20)
    dbn = DBN([32 * 24, 588, 588], cls_count)
    dbn.train(X_mnb, y_mnb, [
        {'epochs': 50, 'eps': 0.05, 'spars': 0.05, 'spars_cost': 0.3},
        {'epochs': 1, 'eps': 0.05}
    ])


def test_dbn_mlp():

    log.info('Testing DBN mlp training')

    d_train, d_test = wf.get_data()
    job = wf.DbnPretrainJob((
        9, [32 * 24, 588, 500],
        [
            [100, 0.05, True, 1, 0.085, 0.15],
            [50, 0.05, True, 1, 0.0, 0.0]
        ]
    ), *d_train[9])

    assert job.is_done()

    dbn = job.results()[0]

    log.info("Fine tuning")
    mlp = dbn.to_mlp()
    log.info('Will test estimator performance before fine tuning')
    analysis.eval_estimator(mlp, 9, 'DBN before fine tuning')
    X_mnb, y_mnb = d_train[9]
    mlp.train(X_mnb, y_mnb, 1, 0.1)

    log.info('Will test estimator performance before after tuning')
    analysis.eval_estimator(mlp, 9, 'DBN as MLP, after fine tuning')


def main():
    logging.basicConfig(level=logging.INFO)
    test_dbn_mlp()


if __name__ == '__main__':
    main()
