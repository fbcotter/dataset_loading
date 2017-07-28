from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pickle

# Package imports
from dataset_loading import core


def load_cifar_data(data_dir, cifar10=True, val_size=2000):
    """Load cifar10 or cifar100 data

    Parameters
    ----------
    data_dir : str
        Path to the folder with the cifar files in them. These should be the
        python files as downloaded from `cs.toronto`__

        __ https://www.cs.toronto.edu/~kriz/cifar.html
    cifar10 : bool
        True if cifar10, false if cifar100
    val_size : int
        Size of the validation set.
    """
    if cifar10:
        train_files = ['data_batch_'+str(x) for x in range(1,6)]
        train_files = [os.path.join(data_dir, f) for f in train_files]
        test_files = ['test_batch']
        test_files = [os.path.join(data_dir, f) for f in test_files]
        num_classes = 10
        label_func = lambda x: np.array(x['labels'], dtype='int32')
    else:
        train_files = ['train']
        train_files = [os.path.join(data_dir, f) for f in train_files]
        test_files = ['test']
        test_files = [os.path.join(data_dir, f) for f in test_files]
        num_classes = 100
        label_func = lambda x: np.array(x['fine_labels'], dtype='int32')

    def load_files(filenames):
        data = np.array([])
        labels = np.array([])
        for name in filenames:
            with open(name, 'rb') as f:
                mydict = pickle.load(f, encoding='latin1')
            newlabels = label_func(mydict)
            data = np.vstack([data, mydict['data']]) if data.size else mydict['data']  # noqa
            labels = np.hstack([labels, newlabels]) if labels.size else newlabels      # noqa
        data = np.reshape(data, [-1, 3, 32, 32], order='C')
        data = np.transpose(data, [0, 2, 3, 1])
        labels = core.convert_to_one_hot(labels, num_classes=num_classes)
        return data, labels

    train_data, train_labels = load_files(train_files)
    test_data, test_labels = load_files(test_files)
    train_data, val_data = np.split(train_data,
                                    [train_data.shape[0]-val_size])
    train_labels, val_labels = np.split(train_labels,
                                        [train_labels.shape[0]-val_size])

    return train_data, train_labels, test_data, test_labels, val_data, val_labels   # noqa


def get_cifar_queues(data_dir, cifar10=True, val_size=2000, transform=None,
                     max_qsize=1000, num_threads=(2,2,2),
                     max_epochs=float('inf'), get_queues=(True, True, True),
                     _rand_data=False):
    """ Get Image queues for CIFAR

    CIFAR10/100 are both small datasets. This function loads them both into
    memory and creates several :py:class:`~dataset_loading.core.ImgQueue`
    instances to feed the training, testing and validation data through to the
    main function. Preprocessing can be done by providing a callable to the
    transform parameter. Note that by default, the CIFAR images returned will be
    of shape [32, 32, 3] but this of course can be changed by the transform
    function.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the cifar data. For cifar10, this should
        be the path to the folder called 'cifar-10-batches-py'. For
        cifar100, this should be the path to the folder 'cifar-100-python'.
    cifar10 : bool
        True if we are using cifar10.
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
    max_qsize : int or tuple of 3 ints
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

    if type(max_qsize) is tuple or type(max_qsize) is list:
        assert len(max_qsize) == 3
        train_qsize, test_qsize, val_qsize = max_qsize
    else:
        train_qsize = max_qsize
        test_qsize = max_qsize
        val_qsize = max_qsize

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
            load_cifar_data(data_dir, cifar10, val_size)
    else:
        # Randomly generate some image like data
        tr_data = np.random.randint(255, size=(10000-val_size, 32, 32, 3))
        tr_labels = np.random.randint(10, size=(10000-val_size,))
        te_data = np.random.randint(255, size=(1000, 32, 32, 3))
        te_labels = np.random.randint(10, size=(1000,))
        val_data = np.random.randint(255, size=(val_size, 32, 32, 3))
        val_labels = np.random.randint(10, size=(val_size,))
        # convert to one hot
        tr_labels = core.convert_to_one_hot(tr_labels)
        te_labels = core.convert_to_one_hot(te_labels)
        val_labels = core.convert_to_one_hot(val_labels)

    # Create the 3 queues
    train_queue = None
    test_queue = None
    val_queue = None
    if get_queues[0]:
        train_queue = core.ImgQueue(maxsize=train_qsize,
                                    name='CIFAR Train Queue')
        train_queue.take_dataset(tr_data, tr_labels, True, train_threads,
                                 train_xfm, max_epochs)
    if get_queues[1]:
        test_queue = core.ImgQueue(maxsize=test_qsize,
                                   name='CIFAR Test Queue')
        test_queue.take_dataset(te_data, te_labels, True, test_threads,
                                test_xfm, max_epochs)
    if get_queues[2] and val_data.size > 0:
        val_queue = core.ImgQueue(maxsize=val_qsize,
                                  name='CIFAR Val Queue')
        val_queue.take_dataset(val_data, val_labels, True, val_threads,
                               val_xfm, max_epochs)

    return train_queue, test_queue, val_queue
