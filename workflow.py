"""
A workflow for running RBM / DBN training
in batches.

The idea is to make a queue of RBMs to train,
and let it all run. Each queue job is stored
when done, so when the queue is interrupted
training progress is lost only for the current
job.
"""

import logging
import util
from rbm import RBM
from dbn import DBN
import numpy as np
import os
import abc


log = logging.getLogger(__name__)

#   the directory where workflow results are stored
DIR = 'workflow_results' + os.sep
DIR_IMG = DIR + 'img' + os.sep

#   ensure that the directories for storing workflow results exist
if not os.path.exists(DIR):
    os.makedirs(DIR)

if not os.path.exists(DIR_IMG):
    os.makedirs(DIR_IMG)


class Job(object):
    """
    A single job to perform. Abstract class for concrete
    RBM and DBN jobs. Defines interface and takes care of result storing.
    """

    def __init__(self, file_name_base):
        """
        Init for Job.

        :param file_name_base: A string defining a file name
            (without folders or extension) under which job
            results should be stored.
        """
        self.file_name_base = file_name_base

        self.results = util.unpickle_unzip(
            DIR + self.file_name_base + '.zip')

    def is_done(self):
        return self.results is not None

    def perform(self):

        self.results = self._perform()
        util.pickle_zip(self.results,
                        DIR + self.file_name_base + '.zip')

    @abc.abstractmethod
    def _perform(self):
        """
        Performs the job and returns the results.
        """
        pass

    def __str__(self):
        return self.file_name_base

    def __repr__(self):
        return self.file_name_base


class RbmJob(Job):

    """
    A Job for traning a single RBM.
    """

    @classmethod
    def params_to_string(cls, params):
        return 'RBM {:02d}_class {:03d}_n_vis {:03d}_n_hid '\
            '{:03d}_epoch {:.3f}_eps {:b}_pcd {:02d}_steps {:.3f}_spars '\
            '{:.3f}_spars_cost'.format(*params)

    def __init__(self, params, train_data):
        """
        Params for RBM training are in a tuple as follows:
        (cls_count, n_vis, n_hid, epochs, epsilon, pcd, steps,
            spars, spars_cost)
        """
        self.params = params
        self.train_data = train_data

        #   job results are stored in the file, attempt to load it
        file_name_base = RbmJob.params_to_string(params)

        super(RbmJob, self).__init__(file_name_base)

    def _perform(self):

        p = self.params
        rbm = RBM(n_vis=p[1], n_hid=p[2])
        cost, time, hid_act = rbm.train(self.train_data, *self.params[3:])

        util.display_RBM(
            rbm, 32, 24, onscreen=False,
            image_file_name=DIR_IMG + self.file_name_base + '.png')

        return (rbm, cost, time, hid_act)


class DbnJob(Job):

    """
    A Job for traning a single DBN.
    """

    @classmethod
    def params_to_string(cls, params):
        r_val = 'DBN {:d}_class {:s}_layers'.format(
            params[0], "_".join([str(x) for x in params[1]]))

        for rbm_ind, rbm_param in enumerate(params[2]):
            r_val += ' RBM{:d}_{:03d}_epoch {:.3f}_eps {:b}_pcd '\
                '{:02d}_steps {:.3f}_spars {:.3f}_spars_cost'.format(
                    rbm_ind, *rbm_param)

        return r_val

    def __init__(self, params, X_train, y_train):
        """
        Params for DBN training are in a tuple as follows:
        (cls_count, layer_size_array, rbm_params), where
        rbm_params is an iterable containing one tuple
        per RBM. The tuple consists of RBM training params:
        (epochs, epsilon, pcd, steps, spars, spars_cost)
        """
        self.params = params
        self.X_train = X_train
        self.y_train = y_train

        #   job results are stored in the file, attempt to load it
        file_name_base = DbnJob.params_to_string(params)

        super(DbnJob, self).__init__(file_name_base)

    def _perform(self):

        p = self.params
        class_count = p[0]
        layer_sizes = p[1]
        rbm_params = p[2]

        dbn = DBN(layer_sizes, class_count)

        #   train the first RBM as an RbmJob (to be able to reuse)
        #   assuming the RBN is more then one layer deep
        if len(layer_sizes) >= 2:
            first_rbm_params = [class_count, layer_sizes[0], layer_sizes[1]]
            first_rbm_params.extend(rbm_params[0])
            first_rbm_job = RbmJob(first_rbm_params, self.X_train)
            if not first_rbm_job.is_done():
                first_rbm_job.perform()

            #   create the DBN, then replace the first
            #   RBM with the one already trained
            dbn.rbms[0] = first_rbm_job.results[0]

            #   make sure it's skipped in DBN training
            rbm_params[0] = None

        train_res = dbn.train(self.X_train, self.y_train, rbm_params)

        return (dbn, train_res)


#   Raw data from the trainset
#   Cached here to avoid duplicate loading.
raw_data = util.load_trainset()


#   a gloabla variable that holds the data returned by
#   the get_data() method. lazily initialized
__data = None


def get_data():
    """
    Returns the data for the workflow: a tuple of two
    dicts (data_train, data_test). Each dictionary maps class
    counts (integers indicating how many classes are used)
    to a tuple of form (X, y) where X are data samples and
    y are labels. Both are numpy arrays.

    The data is lazily initialized into the global __data variable.
    """

    global __data

    if __data is None:

        X, y, classes = raw_data
        log.info('Read %d samples', len(y))

        def data_subset(cls_count):

            cls = ['A', 'B', 'C', 'D', 'E', 'F', 'X', '_BLANK', '_UNKNOWN']
            cls_subs = cls[:cls_count]
            log.info('Taking a subset of data containing classes %r',
                     cls_subs)

            bool_mask = np.array([(classes[ind] in cls_subs) for ind in y])
            X_subs = X[bool_mask]
            y_subs = y[bool_mask]
            log.info('Subset has %d elements', len(X))

            return X_subs, y_subs

        #   splitting the trainset into train / test
        test_size = 0.1
        test_indices = np.array(
            np.random.binomial(1, test_size, len(X)), dtype=np.bool)
        train_indices = np.logical_not(test_indices)

        #   create dicts of data subsets. each dict has form:
        #   {class_count: (X, y)}
        #   note that X and y are for 'train' data split into
        #   minibatches, but for 'test' they are not
        data_train = {}
        data_test = {}
        for cls_cnt in [1, 3, 7, 9]:

            #   get data subset
            X_subs, y_subs = data_subset(cls_cnt)
            N = len(X_subs)

            #   split data subset into train and test
            X_subs_train = X_subs[train_indices[:N]]
            y_subs_train = y_subs[train_indices[:N]]
            X_subs_test = X_subs[test_indices[:N]]
            y_subs_test = y_subs[test_indices[:N]]

            data_train[cls_cnt] = util.create_minibatches(
                X_subs_train, y_subs_train, 20 * cls_cnt)
            data_test[cls_cnt] = (X_subs_test, y_subs_test)

        __data = (data_train, data_test)

    return __data


#   a gloabal variable that holds the job_queue list
#   returned by the job_queue() function
__job_queue = None


def job_queue():
    """
    Returns a list of jobs (some done, some possibly not).
    """

    #   lazy init of a global variable
    global __job_queue
    if __job_queue is not None:
        return __job_queue

    d_train, d_test = get_data()

    #   jobs, params for RBM training are:
    #   (classes, n_vis, n_hid, epochs, epsilon, pcd, steps, spars, spars_cost)
    n_vis = 32 * 24
    # R = RbmJob
    D = DbnJob
    __job_queue = (
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 0.5), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 0.1), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 0.005), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 0.5), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 0.1), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 0.005), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 0.5), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 0.1), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 0.005), d_train[1][0]),

        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 0.075), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 0.025), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 0.1), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 0.025), d_train[3][0]),

        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.075, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.075, 0.075), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.125, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.125, 0.075), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, True, 2, 0.125, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, True, 2, 0.125, 0.075), d_train[3][0]),

        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.10, 0.15), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.10, 0.3), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.075, 0.15), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.075, 0.3), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.05, 0.15), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.05, 0.3), d_train[9][0]),

        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.10, 0.15), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.10, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 2, 0.10, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.075, 0.15), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.075, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.05, 0.15), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.05, 0.3), d_train[9][0]),

        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.10, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.075, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 1, 0.05, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.05, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 4, 0.05, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 15, 0.05, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.035, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.065, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.0, 0.0), d_train[9][0]),

        #   some DBN training!
        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.05],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.085, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        D((
            9, [n_vis, 588, 500],
            [
                [100, 0.05, True, 1, 0.085, 0.15],
                [50, 0.05, True, 1, 0.0, 0.0]
            ]
        ), *d_train[9]),

        D((
            9, [n_vis, 588, 500],
            [
                [30, 0.05, True, 1, 0.085, 0.15],
                [50, 0.05, True, 1, 0.0, 0.0]
            ]
        ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.085, 0.05],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.1, 0.05],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.1, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.15, 0.05],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.15, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.0, 0.0],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     7, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[7]),

        # D((
        #     7, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.3],
        #         [100, 0.03, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[7]),

        # D((
        #     7, [n_vis, 588, 1000],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[7])

    )

    return __job_queue


def main():

    logging.basicConfig(level=logging.DEBUG)
    log.info('Workflow main()')

    for job in job_queue():
        log.info('Evaluating job: %s', job)
        if not job.is_done():
            job.perform()


if __name__ == '__main__':
    main()
