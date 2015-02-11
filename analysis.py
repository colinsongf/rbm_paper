"""
Provides means to analyze models
(RBMs, DBNs, MLPs...).
"""
import numpy as np
from PIL import Image
import math
import logging
import workflow as wf


log = logging.getLogger(__name__)


def display_array(a, dim_y, dim_x, scale=False):
    """
    Displays an array.

    :param a: The array to display.
    :param dim_y: Height of the image (pixels).
    :param dim_x: Width of the image (pixels).
    :param scale: If array values should be scaled
        to the [0, 1] interval, or displayed 'as is'
    """

    a = np.array(a, copy=True, dtype=np.float)

    if scale:
        a -= a.min()
        a /= max(a.max(), 0.000001)
        a *= 255
        a = np.array(a, dtype=np.uint8)

    a = a.reshape((dim_y, dim_x)) * 255

    img = Image.fromarray(a)
    img.show()


def display_weights(W, dim_y, dim_x, ratio=1.333,
                    onscreen=True, image_title='weights',
                    image_file_name=None):
    """
    Displays the visualization of neural net (RBM) weights,
    and optionally saves the image. Weights are grouped per
    hidden neuron, each hidden neuron then displayed as a rectangle
    of dim_y height and dim_x width. Those rectangles are then
    arranged into a grid. This representation is typically
    useful for the first layer of weights in an RBM that works
    with images.

    :param W: The weight matrix to be displayed. Of dimensions
        (n_vis, n_hid), where n_vis is the dimensionality of
        the lower, visible layer that W connects, and is equal
        to dim_y * dim_x.

    :param dim_y: Height of the image represented by the
        visible layer.

    :param dim_x: Width of the image represented by the
        visible layer.

    :param ratio: The desired ratio of the resulting image.
        Note that it might be impossible to get exactly the desired
        ratio, depending on dim_x, dim_y and the number of images.

    :param onscreen: If the image should be displayed onscreen.

    :param image_title: Name to be displayed with the displayed image.

    :param image_file_name: Name of the file where the image should
        be stored. If None, the image is not stored.
    """
    log.info('Displaying weights')

    n_vis, n_hid = W.shape

    #   compose the hidden unit weights into a single image
    #   we use an array that will hold all the pixels of all hiddden / visible
    #   weights, plus a single pixel line between and around

    #   calculate the number of rows and and columns required to get
    #   the desired ratio in an image displaying all hidden units
    rows = (dim_x * n_hid / (ratio * dim_y)) ** 0.5
    cols = int(math.ceil(ratio * dim_y * rows / dim_x))
    rows = int(math.ceil(rows))
    if (rows - 1) * cols >= n_hid:
        rows -= 1
    if (cols - 1) * rows >= n_hid:
        cols -= 1

    margin = 3
    img_array = np.ones((rows * dim_y + (rows + 1) * margin,
                         cols * dim_x + (cols + 1) * margin),
                        dtype=np.uint8)
    img_array *= 128

    #   normalize so that the weight mean is at 0.5
    scale = max(W.max() - W.mean(), W.mean() - W.min()) * 2.0
    W = W * 255 / scale
    W = W - W.min()

    #   iterate though the hidden units
    for hid_ind, hid_weights in enumerate(W.T):

        # get the position of the hidden unit in the image grid
        row_ind = math.floor(hid_ind / cols)
        col_ind = math.floor(hid_ind % cols)

        #   map the hidden unit weights into image
        y = row_ind * (dim_y + margin) + margin
        x = col_ind * (dim_x + margin) + margin
        img_array[y:y + dim_y, x:x + dim_x] = hid_weights.reshape(
            (dim_y, dim_x))

    image = Image.fromarray(img_array)

    #   show image onscren
    if onscreen:
        image.show(title=image_title)

    #   if given the image file name, save it to disk
    if image_file_name is not None:
        image.save(image_file_name)


def display_RBM(rbm, dim_y, dim_x, ratio=1.333,
                onscreen=True, image_title='RBM',
                image_file_name=None):
    """
    Displays the visualization of features of the given rbm on screen,
    and optionally saves the image. A feature is a set of weights
    from a single hidden unit to all the visible ones. features
    are aranged into rows and columns.

    Useful for RBMs that work with images.

    :param rbm: The RBM for which features are to be visualized.

    :param dim_y: Height of the image (visible layer).

    :param dim_x: Width of the image (visible layer).

    :param ratio: The desired ratio of the resulting image.
        Note that it might be impossible to get exactly the desired
        ratio, depending on dim_x, dim_y and the number of images.

    :param onscreen: If the image should be displayed onscreen.

    :param image_title: Name to be displayed with the displayed image.

    :param image_file_name: Name of the file where the image should
        be stored. If None, the image is not stored.
    """
    log.info('Displaying RBM')
    display_weights(rbm.W.get_value(), dim_y, dim_x, ratio,
                    onscreen, image_title, image_file_name)

    #   now handle visible biases
    bias_img_array = rbm.b_vis.get_value().copy().reshape((dim_y, dim_x))
    bias_img_array -= bias_img_array.min()
    bias_img_array /= max(bias_img_array.max(), 0.000001)
    bias_img_array *= 255
    bias_img_array = np.array(bias_img_array, dtype=np.uint8)
    bias_img = Image.fromarray(bias_img_array)
    if onscreen:
        bias_img.show(title=image_title)


def eval_estimator(estimator, class_count=9, name='Unknown_estimator',
                   display_confusion=False):
    """
    Evaluates a single estimator performance on the
    ZEMRIS dataset.

    :type estimator: A estimator object (DBN, MLP, LogReg).
    :param estimator: The estimator to evaluate.

    :type class_count: int
    :param class_count: The number of classes this
        estimator is trained on.

    :type dbn_name: String
    :param dbn_name: Name / description of the estimator.
    """
    _, d_test = wf.get_data()

    X_test, y_test = d_test[class_count]
    y_test_pred = estimator.predict(X_test)

    acc = sum(y_test == y_test_pred) / float(len(y_test))
    f1_macro = f_macro(y_test, y_test_pred)
    log.info('\nEstimator: %r', name)
    log.info('\tacc: %.4f, f1_macro: %.4f', acc, f1_macro)
    if display_confusion:
        log.info('\tConfusion matrix:\n%r',
                 confusion_matrix(y_test, y_test_pred))


def eval_estimator_job_batch(jobs):
    """
    Allows easy estimation of a batch of jobs.
    """
    log.info("Evaluating fitted estimators")

    #   first prepare a list of estimator jobs
    estimator_jobs = []
    for job in jobs:
        if isinstance(job, wf.RbmJob):
            continue

        if isinstance(job, wf.DbnMlpJob):
            estimator_jobs.append(job.pretraining_job())

        estimator_jobs.append(job)

    #   now evaluate those jobs
    for job in estimator_jobs:
        model = job.results()[0]
        eval_estimator(model, name=str(job))


def rbm_hid_act_per_cls(rbm, class_count=9, name='Unknown_RBM'):
    """
    Compiles info about hidden unit activations of an RBM.
    Info is presented in log output,
    as well as visualisation of first features.
    Done only with testing data.
    """

    log.info('RBM hidden layer activation per-class analysis')

    _, d_test = wf.get_data()
    _, _, classes = wf.raw_data

    X_test, y_test = d_test[class_count]

    log.info('Weight mean %.2f', rbm.W.get_value().mean())
    log.info('Hidden layer mean bias %.2f', rbm.b_hid.get_value().mean())

    #   for all the classes get activations
    for cls_ind in range(len(classes)):

        data = X_test[y_test == cls_ind]
        hid_prb, _ = rbm.hid_given_vis(data)
        log.info('Class: %s, data mean: %.3f, hidden act mean: %.4f',
                 classes[cls_ind], data.mean(), hid_prb.mean())

        #   generate the weight matrix that takes into account act_probs
        hid_prb = hid_prb.mean(axis=0)
        hid_prb = hid_prb / hid_prb.max()
        W = rbm.W.get_value().copy() * hid_prb.reshape((1, rbm.n_hid))
        file_name = wf.DIR_IMG + name + ' class_' + classes[cls_ind] + '.png'
        display_weights(W, 32, 24, onscreen=False, image_file_name=file_name)


def acc(truth, prediction):
    """
    Accuracy calculation.

    :param truth: An iterable of integers indicating
        true classes.
    :param prediction: An iterable of integers indicating
        predicted classes.
    """

    sum = 0
    for t, p in zip(truth, prediction):
        if t == p:
            sum += 1

    return sum / float(len(truth))


def f_macro(truth, prediction, classes=None, beta=1.0, return_pr=False):
    """
    Calculates the F-macro measure (averaged over classes)
    for a given set of truth / prediction class label indices.
    Returns the averaged F-score if return_pr parameter
    is False, or a tuple (F_score, precision, recall) if
    return_pr parameter is True.

    :param truth: An iterable of integers indicating
        true classes.
    :param prediction: An iterable of integers indicating
        predicted classes.
    :param classes: An iterable of ints indicating which
        classes should be considered in calculating scores.
        If None (default), then classes in [0, max(labels)]  are
        considered.
    :param beta: Beta parameter of the f-measure. Default
        values is 1.0.
    :param return_pr: If precision and recall scores should
        be returned as well. If False (default), then only
        the F-score is returned, otherwise a tuple containing
        (f-score, precision, recall).
    """

    assert(len(truth) == len(prediction))

    #   get the classes being consider in scoring
    if classes is None:
        classes = range(max(max(truth), max(prediction)) + 1)

    #   for each class calculate everything
    p_scores = []
    r_scores = []
    f_scores = []
    for cls in classes:

        TP, FP, TN, FN = 0, 0, 0, 0
        for t, p in zip(truth, prediction):
            if t == cls:
                if p == cls:
                    TP += 1
                else:
                    FN += 1
            else:
                if p == cls:
                    FP += 1
                else:
                    TN += 1

        if TP == 0:
            precision = 0.
            recall = 0.
            f_score = 0.
        else:
            precision = TP / float(TP + FP)
            recall = TP / float(TP + FN)
            f_score = (1.0 + beta ** 2) * precision * recall \
                / (beta ** 2 * precision + recall)
        p_scores.append(precision)
        r_scores.append(recall)
        f_scores.append(f_score)

    if return_pr:
        return (np.mean(f_scores), np.mean(p_scores), np.mean(r_scores))
    else:
        return np.mean(f_scores)


def confusion_matrix(truth, prediction):
    """
    Calculates and returns the confusion matrix given
    the truth and prediction vectors.

    :param truth: An iterable of integers indicating
        true classes.
    :param prediction: An iterable of integers indicating
        predicted classes.
    """
    assert(len(truth) == len(prediction))

    #   get the number of classes
    cls_count = max(max(truth), max(prediction)) + 1

    matrix = np.zeros((cls_count, cls_count))
    for t, p in zip(truth, prediction):
        matrix[t, p] += 1

    return matrix


def histogram(unit_prb, buckets=50):
    """
    Histograms unit (neuron) activations.

    :param unit_prb: An iterable of probabilities
        of unit activations (one float per unit).
    :param buckets: An int indicating how many buckets
        there should be in the histogram.
    """
    hist = np.zeros(buckets, dtype=float)
    for prb in unit_prb:
        hist[min(buckets - 1, int(prb * buckets))] += 1

    return hist / len(unit_prb)
