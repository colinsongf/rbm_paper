"""
A workflow for running RBM / DBN training
in batches. Goes through a queue
of jobs, performing them one by one,
detecting if they are done, and storing
the results.

The idea is to make a queue of RBMs to train,
and let it all run. Each queue job is stored
when done, so when the queue is interrupted
training progress is lost only for the current
job.
"""

import logging
import util
from rbm import RBM
import numpy as np
import os
import abc


log = logging.getLogger(__name__)


class Job(object):
    """
    A single job to perform. Abstract class for concrete
    RBM and DBN jobs. Takes care of result storing.
    """

    DIR = 'workflow_results' + os.sep
    DIR_IMG = DIR + 'img' + os.sep

    def __init__(self, file_name_base):
        """
        Init for Job.

        :param file_name_base: A string defining a file name
            (without folders or extension) under which job
            results should be stored.
        """
        self.file_name_base = file_name_base

        if not os.path.exists(Job.DIR):
            os.makedirs(Job.DIR)

        if not os.path.exists(Job.DIR_IMG):
            os.makedirs(Job.DIR_IMG)

        self.results = util.unpickle_unzip(
            Job.DIR + self.file_name_base + '.zip')

    def is_done(self):
        return self.results is not None

    def perform(self):

        self.results = self.__perform
        util.pickle_zip(self.results,
                        Job.DIR + self.file_name_base + '.zip')

    @abc.abstractmethod
    def __perform(self):
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

    def __init__(self, params, train_data):
        """
        Params for RBM training are in a tuple as follows:
        (cls_count, n_vis, n_hid, epochs, epsilon, pcd, steps,
            spars, spars_cost)
        """
        self.params = params
        self.train_data = train_data

        #   job results are stored in the file, attempt to load it
        file_name_base = 'RBM {:02d}_class {:03d}_n_vis {:03d}_n_hid '\
            '{:03d}_epoch {:.3f}_eps {:b}_pcd {:02d}_steps {:.3f}_spars '\
            '{:.1f}_spars_cost'.format(*params)

        super(RbmJob, self).__init__(file_name_base)

    def __perform(self):

        p = self.params
        rbm = RBM(n_vis=p[1], n_hid=p[2])
        cost, time, hid_act = rbm.train(self.train_data, *self.params[3:])

        util.display_RBM(
            rbm, 32, 24, onscreen=False,
            image_file_name=Job.DIR_IMG + self.file_name_base + '.png')

        return (rbm, cost, time, hid_act)


def job_queue():

    X, y, classes = util.load_trainset()
    log.info('Read %d samples', len(y))

    def data_subset(subset):
        """
        Creates a subset of data that contains
        only the given classes.
        """

        log.info('Taking a subset of training data for classes: %r', subset)
        bool_mask = np.array([(classes[ind] in subset) for ind in y])
        y_mod = y[bool_mask]
        X_mod = X[bool_mask]
        log.info('Subset has %d elements', len(X_mod))

        return X_mod, y_mod

    X_a, _ = data_subset('A')
    X_abc, _ = data_subset('ABC')

    #   minibatches
    X_mnb = util.create_minibatches(X, None, len(classes) * 20)
    X_a_mnb = util.create_minibatches(X_a, None, 20)
    X_abc_mnb = util.create_minibatches(X_abc, None, 60)

    #   jobs, params for RBM training are:
    #   (classes, n_vis, n_hid, epochs, epsilon, pcd, steps, spars, spars_cost)
    n_vis = 32 * 24
    R = RbmJob
    job_queue = (
        R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 10.0), X_a_mnb),
        R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 2.0), X_a_mnb),
        R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 0.1), X_a_mnb),
        R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 10.0), X_a_mnb),
        R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 2.0), X_a_mnb),
        R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 0.1), X_a_mnb),
        R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 10.0), X_a_mnb),
        R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 2.0), X_a_mnb),
        R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 0.1), X_a_mnb),

        R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 5.0), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 1.5), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 0.5), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 5.0), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 2.0), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 0.5), X_abc_mnb),

        R((3, n_vis, 276, 50, 0.05, False, 1, 0.075, 5.0), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, False, 1, 0.075, 1.5), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, False, 1, 0.125, 5.0), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, False, 1, 0.125, 1.5), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, True, 2, 0.125, 5.0), X_abc_mnb),
        R((3, n_vis, 276, 50, 0.05, True, 2, 0.125, 1.5), X_abc_mnb),

        R((len(classes), n_vis, 432, 50, 0.05, False, 1, 0.10, 3.0), X_mnb),
        R((len(classes), n_vis, 432, 50, 0.05, False, 1, 0.10, 6.0), X_mnb),
        R((len(classes), n_vis, 432, 50, 0.05, False, 1, 0.075, 3.0), X_mnb),
        R((len(classes), n_vis, 432, 50, 0.05, False, 1, 0.075, 6.0), X_mnb),
        R((len(classes), n_vis, 432, 50, 0.05, False, 1, 0.05, 3.0), X_mnb),
        R((len(classes), n_vis, 432, 50, 0.05, False, 1, 0.05, 6.0), X_mnb),

        R((len(classes), n_vis, 588, 50, 0.05, False, 1, 0.10, 3.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, False, 1, 0.10, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, False, 2, 0.10, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, False, 1, 0.075, 3.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, False, 1, 0.075, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, False, 1, 0.05, 3.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, False, 1, 0.05, 6.0), X_mnb),

        R((len(classes), n_vis, 588, 50, 0.05, True, 2, 0.10, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, True, 2, 0.075, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, True, 1, 0.05, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, True, 2, 0.05, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, True, 4, 0.05, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, True, 15, 0.05, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, True, 2, 0.035, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, True, 2, 0.065, 6.0), X_mnb),
        R((len(classes), n_vis, 588, 50, 0.05, True, 2, 0.0, 0.0), X_mnb),
    )

    return job_queue


def main():

    for job in job_queue():
        log.info('Evaluating job: %s', job)
        if not job.is_done():
            job.perform()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
