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
import theano
from time import time
from datetime import date

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


def cost_minimization(inputs, cost, params, epochs, eps, X_mnb, y_mnb):
    """
    Generic cost minimization function (gradient descent) for a
    situaition where given input there are desired outputs.

    :type inputs: iterable of Theano symbolic vars, 2 elements.
    :param inputs: Symblic variables that are inputs to the cost function.
        The iterable needs to consist of two elements, the first is a sym
        variable for minibatch input (X), and the second is a sym for
        minibatch outputs (y).

    :type cost: Theano symbolic variable.
    :param cost: The cost function which needs to be minimized.

    :type params: iterable of theano symbolic vars
    :param params: All the parameters which need to be optimized with
        gradient descent.

    :type epochs: int
    :param epochs: Number of epochs (int) of training.

    :type eps: float
    :param eps: Learning rate.

    :param X_mnb: Trainset split into minibatches. Thus,
        X_mnb is an iterable containing numpy arrays of
        (mnb_N, n_vis) shape, where mnb_N is the number of
        samples in the minibatch.

    :param y_mnb: Trainset label indices split into minibatches. Thus,
        y_mnb is an iterable containing numpy arrays of
        (mnb_N, ) shape, where mnb_N is the number of
        samples in the minibatch.
    """

    #   gradients and param updates
    grads = [(p, theano.tensor.grad(cost=cost, wrt=p)) for p in params]
    updates = [(p, p - eps * grad_p) for (p, grad_p) in grads]

    # compiled training function
    train_model = theano.function(
        inputs=inputs,
        outputs=cost,
        updates=updates
    )

    #   things we'll track through training, for reporting
    epoch_costs = []
    epoch_times = []

    #   iterate through the epochs
    for epoch in range(epochs):
        log.info('Starting epoch %d', epoch)
        epoch_t0 = time()

        #   iterate through the minibatches
        batch_costs = []
        for batch_ind, (X_batch, y_batch) in enumerate(zip(X_mnb, y_mnb)):
            batch_costs.append(train_model(X_batch, y_batch))

        epoch_costs.append(np.array(batch_costs).mean())
        epoch_times.append(time() - epoch_t0)
        log.info(
            'Epoch cost %.5f, duration %.2f sec',
            epoch_costs[-1],
            epoch_times[-1]
        )

    log.info('Training duration %.2f min',
             (sum(epoch_times)) / 60.0)

    return epoch_costs, epoch_times


def write_ndarray(ndarray, file, formatter=None, separators=None):
    """
    Writes a numpy array into a file.

    :param ndarray: The array to write to file.
    :param file: File object in which to write.
    :param formatter: Formatting string to be used on each
        numpy array element if None (default), the '{}' is used.
    :param separators: A list of separator tokens to be used
        in between of array elements.
    """

    shape = ndarray.shape
    #   get cumulative sizes of each dimension
    dim_sizes = [
        np.prod(shape[(i + 1):], dtype=int) for i in range(0, len(shape))]

    #   prepare the separators
    if separators is None:
        separators = [os.linesep] * len(shape)
        separators[-1] = ' '

    #   default formatter
    if formatter is None:
        formatter = "{}"

    #   write all the array elements
    for i, n in enumerate(ndarray.reshape(ndarray.size, )):
        file.write(formatter.format(n))
        sep_ind = [(i + 1) % ds for ds in dim_sizes].index(0)
        file.write(separators[sep_ind])


def store_mlp_ascii(mlp, file_path):
    """
    Stores a MLP into an ASCII file.

    :param mlp: A MLP instance to store.
    :param file_path: File path to store it to.
    """

    log.info("Storing MLP to file: %s", file_path)

    #   first info in the ascii file is the layer sizes
    layer_sizes = [32 * 24]
    for hid_lay in mlp.hidden_layers:
        layer_sizes.append(hid_lay.b.get_value().size)
    layer_sizes.append(mlp.regression_layer.b.get_value().size)

    with open(file_path, "w") as file:

        def ln(string):
            file.write(string + os.linesep)

        ln("# Multilayer-perceptron, exported from Theano+Python DBN-MLP")
        ln("# Author: Florijan Stamenkovic (florijan.stameknovic@gmail.com")
        ln("# Date: {}".format(date.today()))
        ln("#")
        ln("# Non-comment lines are organized as follows:")
        ln("#   - first come layer sizes (visible -> hidden -> softmax")
        ln("#   - then for each layer (except visible):")
        ln("#       - first the weights to previous layer in N lines where N "
            "is number of neurons of previous layer")
        ln("#       - then biases for that layer (in a single line)")
        ln("# Enjoy!!!")

        ln(" ".join([str(ls) for ls in layer_sizes]))

        for hl in mlp.hidden_layers:
            write_ndarray(hl.W.get_value(), file, "{:.06f}")
            write_ndarray(hl.b.get_value(), file, "{:.06f}", [os.linesep])

        write_ndarray(mlp.regression_layer.W.get_value(), file, "{:.06f}")
        write_ndarray(mlp.regression_layer.b.get_value(), file, "{:.06f}")
