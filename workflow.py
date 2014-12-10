"""
Provides infrastructure for defining
job batches, where each "job" is the
traning of a an RBM, a DBN, MLP or whatever.
Also provides data handling.
"""

import logging
import util
import analysis
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
    RBM, DBN, whatever jobs. Defines interface and takes
    care of storing results to the hard drive (zipped pickle).
    """

    def __init__(self):
        """
        Init for Job.
        """

    def results(self):
        if getattr(self, '_results', False) is False:
            self._results = util.unpickle_unzip(
                DIR + self.file_name_base() + '.zip')

        return self._results

    def is_done(self):
        return self.results() is not None

    def perform(self):

        self._results = self._perform()
        util.pickle_zip(self._results,
                        DIR + self.file_name_base() + '.zip')

    @abc.abstractmethod
    def _perform(self):
        """
        Performs the job and returns the results.
        """
        pass

    @abc.abstractmethod
    def file_name_base(self):
        """
        Retruns a unique filename base (no folder nor extension)
        for this job.
        """
        pass

    def __str__(self):
        return self.file_name_base()

    def __repr__(self):
        return self.file_name_base()


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

    def file_name_base(self):
        return 'RBM {:02d}_class {:03d}_n_vis {:03d}_n_hid '\
            '{:03d}_epoch {:.3f}_eps {:b}_pcd {:02d}_steps {:.3f}_spars '\
            '{:.3f}_spars_cost'.format(*self.params)

    def _perform(self):

        p = self.params
        rbm = RBM(n_vis=p[1], n_hid=p[2])
        cost, time, hid_act = rbm.train(self.train_data, *self.params[3:])

        analysis.display_RBM(
            rbm, 32, 24, onscreen=False,
            image_file_name=DIR_IMG + self.file_name_base() + '.png')

        return (rbm, cost, time, hid_act)


class DbnPretrainJob(Job):
    """
    A Job for pretraning a single DBN as a stack of RBMs.
    """

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

    def file_name_base(self):
        r_val = 'DBN_pretrain {:d}_class {:s}_layers'.format(
            self.params[0], "_".join([str(x) for x in self.params[1]]))

        for rbm_ind, rbm_param in enumerate(self.params[2]):
            r_val += ' RBM{:d}_{:03d}_epoch {:.3f}_eps {:b}_pcd '\
                '{:02d}_steps {:.3f}_spars {:.3f}_spars_cost'.format(
                    rbm_ind, *rbm_param)

        return r_val

    def _perform(self):

        p = self.params
        class_count = p[0]
        layer_sizes = p[1]
        #   make a copy of rbm-param list because we might
        #   modify it for training, and we don't want to
        #   affect the original
        rbm_params = list(p[2])

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
            dbn.rbms[0] = first_rbm_job.results()[0]

            #   make sure it's skipped in DBN training
            rbm_params[0] = None

        train_res = dbn.pretrain(self.X_train, self.y_train, rbm_params)

        return (dbn, train_res)


class DbnMlpJob(Job):
    """
    A job for training a deep net by first
    pretraining the net as a DBN, and then fine-tuning
    as a MLP
    """

    def __init__(self, params, X_train, y_train):
        """
        Params for DBN-MLP training are in a tuple as follows:
        (cls_count, layer_size_array, rbm_params, ft_epoch, ft_eps),
        where rbm_params is an iterable containing one tuple
        per RBM. The tuple consists of RBM training params:
        (epochs, epsilon, pcd, steps, spars, spars_cost).
        ft_epoch is the number of fine-tuning epochs and
        ft_eps is the learning rate in fine-tuning.
        """
        self.params = params
        self.X_train = X_train
        self.y_train = y_train

    def file_name_base(self):
        r_val = 'DBN_MLP {:d}_class {:s}_layers'.format(
            self.params[0], "_".join([str(x) for x in self.params[1]]))

        for rbm_ind, rbm_param in enumerate(self.params[2]):
            r_val += ' RBM{:d}_{:03d}_epoch {:.3f}_eps {:b}_pcd '\
                '{:02d}_steps {:.3f}_spars {:.3f}_spars_cost'.format(
                    rbm_ind, *rbm_param)

        r_val += " MLP_FT {:03d}_epoch {:.3f}_eps".format(
            self.params[3], self.params[4])

        return r_val

    def pretraining_job(self):
        """
        Returns the pretraining aspect of this
        job (a DbnPretrainJob instance).
        """
        return DbnPretrainJob(
            self.params[:3], self.X_train, self.y_train)

    def _perform(self):

        fine_tune_epoch = self.params[3]
        fine_tune_eps = self.params[4]

        #   pre-train DBN as a sub-job
        pretrain_job = self.pretraining_job()
        if not pretrain_job.is_done():
            pretrain_job.perform()

        #   fine tuning
        mlp = pretrain_job.results()[0].to_mlp()
        train_res = mlp.train(self.X_train, self.y_train,
                              fine_tune_epoch, fine_tune_eps)

        return (mlp, train_res)


#   Raw data from the trainset
#   Cached here to avoid duplicate loading.
raw_data = util.load_trainset()


#   a gloabla variable that holds the data returned by
#   the get_data() method. lazily initialized
__data = None


def get_data():
    """
    Returns the data for the workflow: a tuple of two
    dicts (data_train, data_test). data_train maps class
    counts (integers indicating how many classes are used)
    to a tuple of form (X_mnb, y_mnb) where X_mnb are data samples
    split into minibatches and y are corresponding labels.
    data_test maps class counts to a tuple of form (X, y) where
    X are data samples and y are corresponding labels, NOT
    split into minibatches.

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
