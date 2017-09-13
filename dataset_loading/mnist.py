# Much of the code to load the MNIST dataset from the zip files was taken from
# the tensorflow source.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import time

# Package imports
from dataset_loading import core, utils
import gzip


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Parameters
    ----------
    f: file object
        file that can be passed into a gzip reader.

    Returns
    -------
    data: A 4D uint8 numpy array [index, y, x, depth].

    Raises
    ------
    ValueError: If the bytestream does not start with 2051.
    """
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Parameters
    ----------
    f: file object
        A file object that can be passed into a gzip reader.
    one_hot: bool
        Does one hot encoding for the result.
    num_classes: int
        Number of classes for the one hot encoding.

    Returns
    -------
    labels: a 1D uint8 numpy array.

    Raises
    ------
    ValueError: If the bystream doesn't start with 2049.
    """
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            labels = utils.convert_to_one_hot(labels, num_classes=num_classes)
        return labels


def load_mnist_data(data_dir, val_size=2000):
    """Load mnist data

    Parameters
    ----------
    data_dir : str
        Path to the folder with the mnist files in them. These should
        be the gzip files downloaded from `yann.lecun.com`__

        __ http://yann.lecun.com/exdb/mnist/
    val_size : int
        Size of the validation set.
    """
    if not 0 <= val_size <= 60000:
        raise ValueError(
            'Validation size should be between 0 and 60000. Received: {}.'
            .format(60000, val_size))

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = os.path.join(data_dir, TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_data = extract_images(f)

    local_file = os.path.join(data_dir, TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=True)

    local_file = os.path.join(data_dir, TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_data = extract_images(f)

    local_file = os.path.join(data_dir, TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=True)

    val_data = train_data[:val_size]
    val_labels = train_labels[:val_size]
    train_data = train_data[val_size:]
    train_labels = train_labels[val_size:]

    return train_data, train_labels, test_data, test_labels, val_data, val_labels   # noqa


def get_mnist_queues(data_dir, val_size=2000, transform=None,
                     maxsize=1000, num_threads=(2,2,2),
                     max_epochs=float('inf'), get_queues=(True, True, True),
                     _rand_data=False):
    """ Get Image queues for MNIST

    MNIST is a small dataset. This function loads it into memory and creates
    several :py:class:`~dataset_loading.core.ImgQueue` to feed the training,
    testing and validation data through to the main function.  Preprocessing can
    be done by providing a callable to the transform parameter.  Note that by
    default, the black and white MNIST images will be returned as a [28, 28, 1]
    shape numpy array. You can of course modify this with the transform
    function.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the cifar data. For cifar10, this should
        be the path to the folder called 'cifar-10-batches-py'. For
        cifar100, this should be the path to the folder 'cifar-100-python'.
    val_size : int
        How big you want the validation set to be. Will be taken from the end of
        the train data.
    transform : None or callable or tuple of callables
        Callable function that accepts a numpy array representing **one** image,
        and transforms it/preprocesses it. E.g. you may want to remove the mean
        and divide by standard deviation before putting into the queue. If tuple
        of callables, needs to be of length 3 and should be in the order
        (train_transform, test_transform, val_transform). Setting it to None
        means no processing will be done before putting into the image queue.
    maxsize : int or tuple of 3 ints
        How big the image queues will be. Increase this if your main program is
        chewing through the data quickly, but increasing it will also mean more
        memory is taken up. If tuple of ints, needs to be length 3 and of the
        form (train_qsize, test_qsize, val_qsize).
    num_threads : int or tuple of 3 ints
        How many threads to use for the train, test and validation threads (if
        tuple, needs to be of length 3 and in that order).
    max_epochs : int
        How many epochs to run before returning FileQueueDepleted exceptions
    get_queues : tuple of 3 bools
        In case you only want to have training data, or training and validation,
        or any subset of the three queues, you can mask the individual queues by
        putting a False in its position in this tuple of 3 bools.

    Returns
    -------
    train_queue : :py:class:`~dataset_loading.core.ImgQueue` instance or None
        Queue with the training data in it. None if get_queues[0] == False
    test_queue : :py:class:`~dataset_loading.core.ImgQueue` instance or None
        Queue with the test data in it. None if get_queues[1] == False
    val_queue : :py:class:`~dataset_loading.core.ImgQueue` instance or None
        Queue with the validation data in it. Will be None if the val_size
        parameter was 0 or get_queues[2] == False

    Notes
    -----
    If the max_epochs paramter is set to a finite amount, then when the queues
    run out of data, they will raise a dataset_loading.FileQueueDepleted
    exception.
    """
    # Process the inputs that can take multiple forms.
    if transform is None:
        train_xfm = None
        test_xfm = None
        val_xfm = None
    else:
        if type(transform) is tuple or type(transform) is list:
            assert len(transform) == 3
            train_xfm, test_xfm, val_xfm = transform
        else:
            train_xfm = transform
            test_xfm = transform
            val_xfm = transform

    if type(maxsize) is tuple or type(maxsize) is list:
        assert len(maxsize) == 3
        train_qsize, test_qsize, val_qsize = maxsize
    else:
        train_qsize = maxsize
        test_qsize = maxsize
        val_qsize = maxsize

    if type(num_threads) is tuple or type(num_threads) is list:
        assert len(num_threads) == 3
        train_threads, test_threads, val_threads = num_threads
    else:
        train_threads = num_threads
        test_threads = num_threads
        val_threads = num_threads

    # Load the data into memory
    if not _rand_data:
        tr_data, tr_labels, te_data, te_labels, val_data, val_labels = \
            load_mnist_data(data_dir, val_size)
    else:
        # Randomly generate some image like data
        tr_data = np.random.randint(255, size=(10000-val_size, 28, 28))
        tr_labels = np.random.randint(10, size=(10000-val_size,))
        te_data = np.random.randint(255, size=(1000, 28, 28))
        te_labels = np.random.randint(10, size=(1000,))
        val_data = np.random.randint(255, size=(val_size, 28, 28))
        val_labels = np.random.randint(10, size=(val_size,))
        # convert to one hot
        tr_labels = utils.convert_to_one_hot(tr_labels)
        te_labels = utils.convert_to_one_hot(te_labels)
        val_labels = utils.convert_to_one_hot(val_labels)

    # Create the 3 queues
    train_queue = None
    test_queue = None
    val_queue = None
    if get_queues[0]:
        train_queue = core.ImgQueue(maxsize=train_qsize,
                                    name='MNIST Train Queue')
        train_queue.take_dataset(tr_data, tr_labels, True, train_threads,
                                 train_xfm, max_epochs)
    if get_queues[1]:
        test_queue = core.ImgQueue(maxsize=test_qsize,
                                   name='MNIST Test Queue')
        test_queue.take_dataset(te_data, te_labels, True, test_threads,
                                test_xfm)
    if get_queues[2] and val_data.size > 0:
        val_queue = core.ImgQueue(maxsize=val_qsize,
                                  name='MNIST Val Queue')
        val_queue.take_dataset(val_data, val_labels, True, val_threads,
                               val_xfm)

    # allow for the filling of the queues with some samples
    time.sleep(0.5)
    return train_queue, test_queue, val_queue
