"""
Utilities for the Deep-Belief-Net Theano implementation.
"""

import logging
import math
import numpy as np
from PIL import Image
from zipfile import ZipFile, ZIP_DEFLATED
import os
from io import BytesIO
import pickle

log = logging.getLogger(__name__)


def create_minibatches(X, y, size, shuffle=True):
    """
    Default implementation for batching the
    data, override for finer control.

    Returns batched data in the form of a
    list of (X, y) batches if y is not None.
    Otherwise, if y is None, it returns a list
    of X batches.

    :type X: list
    :param X: A list of sentences. Each sentence
    is a list of indices of vocabulary terms.

    :type y: list
    :param y: A list of sentence labels, when using
    the RAE in a supervised fashion, or None when purely
    unsupervised.

    :type size: int, float
    :param size: Desired size of the minibatches. If
    int, then taken as the desired size. If float (0, 1), then
    taken as the desired perecentage X.

    :type shuffle: boolean
    :param shuffle: If or not the trainset should be
    shuffled prior to splitting into batches. If the trainset
    is ordered with regards to classes, shuffling will ensure
    that classes are approximately uniformly represented in
    each minibatch.

    """
    #   convert float size to int size
    if isinstance(size, float):
        size = int(math.ceil(len(X) * size))

    #   if size out of range, ensure appropriate
    size = min(size, len(X))
    size = max(1, size)
    log.info('Creating minibatches, size: %d', size)

    #   shuffle trainset
    if shuffle:
        if y is not None:
            assert len(X) == len(y)
            p = np.random.permutation(len(X))
            X = X[p]
            y = y[p]
        else:
            np.random.shuffle(X)

    #   split X and y into a batch of tuples
    batches_X = []
    batches_y = []
    while True:
        low_ind = len(batches_X) * size
        high_ind = min(low_ind + size, len(X))
        batches_X.append(X[low_ind:high_ind])
        if y is not None:
            batches_y.append(y[low_ind:high_ind])

        if high_ind >= len(X):
            break

    log.info('Created %d minibatches', len(batches_X))

    if y is not None:
        return batches_X, batches_y
    else:
        return batches_X


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
    W = rbm.W.get_value()

    #   compose the hidden unit weights into a single image
    #   we use an array that will hold all the pixels of all hiddden / visible
    #   weights, plus a single pixel line between and around

    #   calculate the number of rows and and columns required to get
    #   the desired ratio in an image displaying all hidden units
    rows = (dim_x * rbm.n_hid / (ratio * dim_y)) ** 0.5
    cols = int(math.ceil(ratio * dim_y * rows / dim_x))
    rows = int(math.ceil(rows))
    if (rows - 1) * cols >= rbm.n_hid:
        rows -= 1
    if (cols - 1) * rows >= rbm.n_hid:
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

    #   now handle visible biases
    bias_img_array = rbm.b_vis.get_value().copy().reshape((dim_y, dim_x))
    bias_img_array -= bias_img_array.min()
    bias_img_array /= max(bias_img_array.max(), 0.000001)
    bias_img_array *= 255
    bias_img_array = np.array(bias_img_array, dtype=np.uint8)
    bias_img = Image.fromarray(bias_img_array)
    if onscreen:
        bias_img.show(title=image_title)


def load_trainset():
    """
    Loads the trainset from the default files (attempting
    the prepared pickle, if unavailable then from zip).

    Returns a tuple (X, y, classes) where X is a numpy
    array of shape (N, pixel_count) containing binarized
    images, y is a numpy array of labels as indexes,
    shaped (N, 1), and 'classes' contains the names of classes
    so that an index in y gives the class name in 'classes'.
    """

    log.info('Loading trainset')

    #   attemts to load the trainset from a pickled file
    pickled_file_name = 'Trainset_pickle.zip'
    data = unpickle_unzip(pickled_file_name)
    if data is not None:
        return data

    data = load_trainset_zip('Trainset_raw.zip')

    #   store data for subsequent usage
    pickle_zip(data, pickled_file_name)

    return data


def unpickle_unzip(file_name):
    """
    Looks for a file with the given file_name,
    attempts to unzip it's first entry, and then
    unpickle tne unzipped data, which it returns.

    If it fails at any point, it silently returns None.
    """

    try:
        zip = ZipFile(file_name, 'r')
        file_in_zip = zip.namelist()[0]
        data = pickle.load(BytesIO(zip.read(file_in_zip)))
        log.info('Succesfully loaded zipped pickle')
        return data
    except IOError:
        log.info('Failed to load zipped pickle')
        return None
    finally:
        if 'file' in locals():
            file.close()


def pickle_zip(data, file_name):
    """
    Pickles given data, and then zips the
    result and stores it into a file with
    the given file_name.

    Returns True if succesful, False otherwise.
    """

    try:
        log.info('Attempting to pickle data')
        zip = ZipFile(file_name, 'w', ZIP_DEFLATED)
        zip.writestr('Trainset.pickle', pickle.dumps(data))
        return True
    except IOError:
        log.info('Failed to pickle data')
        return False
    finally:
        if 'file' in locals():
            file.close()


def load_trainset_zip(file_name):
    """
    Loads the trainset from a ZIP file containing them.
    Returns a tuple (X, y, classes) where X is a numpy
    array of shape (N, pixel_count) containing binarized
    images, y is a numpy array of labels as indexes,
    shaped (N, 1), and 'classes' contains the names of classes
    so that an index in y gives the class name in 'classes'.

    :param file_name: Path to the ZIP file.
    """

    log.info('Loading trainset from ZIP file %s', file_name)

    zip_file = ZipFile(file_name, 'r')
    entries = zip_file.namelist()

    #   filter out the non-images from entries
    images = filter(lambda s: s.endswith('.png'), entries)

    #   get class names
    classes = sorted(set([n[:n.index(os.sep)] for n in images]))

    #   get image size from the first image
    size = Image.open(BytesIO(zip_file.read(images[0]))).size

    #   create return values
    X = np.zeros((len(images), size[0] * size[1]), dtype=np.int8)
    y = np.zeros(len(images), dtype=np.int8)

    #   load images
    for img_ind, image in enumerate(images):

        log.debug('Processing image %s', image)
        im = Image.open(BytesIO(zip_file.read(image)))

        X[img_ind][:] = [0 if p < 128 else 1 for p in im.getdata()]
        y[img_ind] = classes.index(image[:image.index(os.sep)])

    return X, y, classes


def labels_to_indices(labels):
    """
    Converts an iterable of labels into
    a numpy vector of label indices (zero-based).

    Returns a tuple (indices, vocabulary) so that
    vocabulary[index]=label
    """
    vocab = sorted(set(labels))
    indices = np.array([vocab.index(lab) for lab in labels], dtype=np.int)

    return indices, vocab


def one_hot(indices, count=None):
    """
    Takes a vector of 0 based indices (numpy array) and
    converts it into a matrix of one-hot-encoded
    indices (each index becomes one row).

    For example, if 'indices' is [2, 3], the
    results is:
    [
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ]

    :param indices: The indices to convert.

    :param count: The number elements each one-hot-encoded
        vector should have. If 'None', it is assumed to be
        (indices.max() + 1)
    """

    #   ensure indices is a vector
    indices = indices.reshape(indices.size)

    #   get the max size
    if count is None:
        count = indices.max() + 1
    else:
        assert indices.max() < count

    encoded = np.zeros((indices.size, count), dtype=np.uint8)
    encoded[range(indices.size), indices] = 1

    return encoded


class __lin_reducer(object):

    """
    Class used by the 'lin_reducer' function.
    """

    def __init__(self, start, step):
        self.start = start
        self.step = step

    def __call__(self, epoch_ind, epoch_errors):
        return self.start - self.step * epoch_ind

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "({:.3f} - {:.5f} * epoch)".format(
            self.start, self.step)


def lin_reducer(start, end, epochs=100):
    """
    Creates and returns a callable that is used
    for calculating epsilon (learning rate) in the RBM
    depending on the epoch and/or previous epoch costs.
    Epsilon is calculated to liearly reduce from
    'start' to 'end' in 'epoch' number of epochs.

    :param start: The desired initial learning rate.

    :param end: The desired learning rate at point
        indicated by 'epochs' param.

    :param epochs: See 'end'.
    """
    return __lin_reducer(start, (start - end) / float(epochs))


def f_macro(truth, prediction, beta=1.0):
    """
    Calculates the F-macro measure (averaged over classes)
    for a given set of truth / prediction class label indices.

    :param truth: An iterable of integers indicating
        true classes.
    :param prediction: An iterable of integers indicating
        predicted classes.
    :param beta: Beta parameter of the f-measure.
    """

    assert(len(truth) == len(prediction))

    #   get the number of classes
    cls_count = max(max(truth), max(prediction)) + 1

    #   for each class calculate everything
    scores = np.zeros(cls_count)
    for cls in range(cls_count):

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

        precision = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        f_score = (1.0 + beta ** 2) * precision * recall \
            / (beta ** 2 * precision + recall)
        scores[cls] = f_score

    return scores.mean()


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
