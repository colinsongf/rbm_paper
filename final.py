"""
A script for final evaluation
of the best known DBN / MLP
classifier for the ZEMRIS dataset.
Also contains the routine for
calculating precision@recall and
displaying it as a graph.
"""

import logging
import os
import numpy as np
from dbn import DBN
import util
import analysis
import matplotlib.pyplot as plt
import math

#   the directory where final results are stored
DIR = 'final_results' + os.sep

#   ensure that the directories for storing workflow results exist
if not os.path.exists(DIR):
    os.makedirs(DIR)

log = logging.getLogger(__name__)

#   the ZEMRIS dataset
__X, __y, __CLASSES = util.load_trainset()
__CLASS_COUNT = len(__CLASSES)

#   TODO remove!!
# print '\n\n\t****** REDUCING DATASET ********\n\n'
# mask = np.random.randint(0, 25, len(__X)) == 0
# __X = __X[mask]
# __y = __y[mask]


def folds(sample_count, fold_count=10):
    """
    Generates fold masks and returns them
    in a list. Each fold mask is a boolean
    numpy array. Folds are generated randomly,
    stratification across classess is not ensured.

    :param sample_count: The number of samples
        that need to be split into folds.
    :param fold_count: The desired number of
        folds.
    """
    log.debug("Generating %d folds for %d samples",
              fold_count, sample_count)

    #   random generator we use to always get the same folds
    rng = np.random.RandomState(67345862)

    #   indices of sample-fold belonging
    indices = rng.randint(0, fold_count, sample_count)

    #   generate and return masks
    return [indices == i for i in range(fold_count)]


def train_classifier(X, y):
    """
    Trains a classifier using best known
    parameters on given data / labels.

    :param X: Samples, a numpy array of
        (N, n_vis) shape where N is number of
        samples and n_vis number of visible
        varliables (sample dimensionality).

    :param y: Labels, a numpy array of
        (N, 1) shape. Each lable should be
        a label index.
    """

    #   split data into minibatches
    X_mnb, y_mnb = util.create_minibatches(X, y, __CLASS_COUNT * 20)

    #   create a DBN and pretrain
    dbn = DBN([32 * 24, 600, 600], __CLASS_COUNT)
    pretrain_params = [
        [80, 0.05, True, 1, 0.085, 0.1],
        [80, 0.05, True, 1, 0.000, 0.0]
    ]
    dbn.pretrain(X_mnb, y_mnb, pretrain_params)

    #   fine-tuning
    mlp = dbn.to_mlp()
    mlp.train(X_mnb, y_mnb, 1000, 0.1)

    return mlp


def plot_precision_recall(
        threshs, precisions, recalls, above_thresh,
        show=True, file_name=None, xlim=[0., 1.], ylim=[0., 1.]):
    """
    Plots precisions and recalls against confidence
    threshholds.

    :param threshs: A numpy array of threshold values.
    :param precisions: A numpy array of precision values.
    :param recalls: A numpy array of precision values.
    :param show: If the plot should be shown onscreen.
    :param file_name: Name of the file to store the plot in.
        If None, the plot image is not stored.
    """
    assert threshs.shape == precisions.shape
    assert threshs.shape == recalls.shape

    #   start plotting
    plt.figure(figsize=(12, 9), dpi=72)

    plt.plot(threshs, precisions, label="Preciznost")
    plt.plot(threshs, recalls, label="Odziv")
    plt.plot(threshs, above_thresh, label="Iznad praga")
    plt.xlim(xlim)
    plt.xticks(np.arange(*xlim, step=(xlim[1] - xlim[0]) / 10.))
    plt.ylim(ylim)
    plt.yticks(np.arange(*ylim, step=(ylim[1] - ylim[0]) / 10.))
    plt.xlabel("Prag pouzdanosti klasifikacije")
    plt.legend(loc=3)
    plt.grid()

    #   show plot or save to file
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()


def evaluate():
    """
    Evaluates the best known classifier on the
    ZEMRIS datased using 10-fold cross validation.
    """

    fold_count = 10
    log.info("Evaluating best classifier using %d-fold cross-val", fold_count)

    #   confidence thresholds, used for calculating precision@recal
    threshs = np.arange(101, dtype=float) * 6. / 100.
    threshs = 1 - math.e ** (-threshs)
    threshs += 1 - threshs[-1]

    #   evaluation data, per fold
    f1_scores = np.zeros(fold_count)
    precisions = np.zeros((fold_count, threshs.size))
    recalls = np.zeros((fold_count, threshs.size))
    above_thresh = np.zeros((fold_count, threshs.size))

    #   iterate over folds
    for fold_ind, fold_mask in enumerate(folds(len(__X), fold_count)):
        log.info("Evaluating fold %d", fold_ind)

        #   attempt to load an already trained classifier for this fold
        classifier_path = "classifier_eval_fold_{:02d}.zip".format(fold_ind)
        classifier_path = os.path.join(DIR, classifier_path)
        classifier = util.unpickle_unzip(classifier_path)
        if classifier is None:
            #   didn't find an existing classifier, so train one
            train_X = __X[np.logical_not(fold_mask)]
            train_y = __y[np.logical_not(fold_mask)]
            classifier = train_classifier(train_X, train_y)
            #   store classifier for the future
            util.pickle_zip(classifier, classifier_path)

        #   evaluate classifier
        test_X = __X[fold_mask]
        test_y = __y[fold_mask]
        #   get confidences for classifier
        test_y_conf = classifier.predict_confidence(test_X)
        #   get classification labels
        test_y_pred = np.argmax(test_y_conf, axis=1)

        #   log the classificaton scores
        f1_score = analysis.f_macro(test_y, test_y_pred)
        conf_matrix = analysis.confusion_matrix(test_y, test_y_pred)
        log.info("\nFold %d F1-macro: %.3f, confusion matrix:\n%r",
                 fold_ind, f1_score, conf_matrix)
        f1_scores[fold_ind] = f1_score

        #   generate the precision-recall curve data
        for i, thresh in enumerate(threshs):
            #   modifiy prediction labels so that
            #   all predictions below the threshold are
            #   set to a wrong label
            thresh_pred = np.array(test_y_pred)
            below_thresh = test_y_conf[
                np.arange(len(test_y_conf)), test_y_pred] < thresh
            thresh_pred[below_thresh] = -1
            _, precision, recall = analysis.f_macro(
                test_y, thresh_pred, range(__CLASS_COUNT), 1.0, True)
            precisions[fold_ind, i] = precision
            recalls[fold_ind, i] = recall
            above_thresh[fold_ind, i] = np.logical_not(below_thresh).mean()

    #   report final data
    log.info("\n\nFinal F1: %.3f +- %.3f", f1_scores.mean(), np.std(f1_scores))
    precisions = precisions.mean(axis=0)
    recalls = recalls.mean(axis=0)
    above_thresh = above_thresh.mean(axis=0)
    plot_precision_recall(threshs, precisions, recalls, above_thresh, True,
                          os.path.join(DIR, "precision_recall_final.pdf"))
    plot_precision_recall(threshs, precisions, recalls, above_thresh, True,
                          os.path.join(DIR, "precision_recall_final_det.pdf"),
                          [0.8, 1.0], [0.9, 1.0])


def train_final():
    """
    Trains the final classifier and returns it.
    """

    log.info("Training final classifier")

    #   attempt to load an already trained classifier for this fold
    classifier_path = os.path.join(DIR, "classifier_final.zip")
    classifier = util.unpickle_unzip(classifier_path)
    if classifier is None:
        #   didn't find an existing classifier, so train one
        classifier = train_classifier(__X, __y)
        #   store classifier for the future
        util.pickle_zip(classifier, classifier_path)

    return classifier


def main():
    logging.basicConfig(level=logging.DEBUG)
    log.info("Final DBN/MLP evaluation for ZEMRIS dataset")

    evaluate()
    classifier = train_final()
    ascii_path = os.path.join(DIR, "classifier_final_ascii.txt")
    util.store_mlp_ascii(classifier, ascii_path)

if __name__ == '__main__':
    main()
