from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from .core import convert_to_one_hot
import pickle


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
            data = np.vstack([data, mydict['data']]) if data.size else mydict['data']
            labels = np.hstack([labels, newlabels]) if labels.size else newlabels
        data = np.reshape(data, [-1, 3, 32, 32], order='C')
        data = np.transpose(data, [0, 2, 3, 1])
        labels = convert_to_one_hot(labels, num_classes=num_classes)
        return data, labels

    train_data, train_labels = load_files(train_files)
    test_data, test_labels = load_files(test_files)
    train_data, val_data = np.split(train_data, [48000])
    train_labels, val_labels = np.split(train_labels, [48000])

    return train_data, train_labels, test_data, test_labels, val_data, val_labels
