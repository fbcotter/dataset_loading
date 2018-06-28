from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import time

# Package imports
from dataset_loading import core, utils


def load_synsets(data_dir=None):
    """ Loads the synset data for the cls-loc dataset.

    Returns
    -------
    synsets : list
        List of structs of length 1860. Each struct has the attributes:

        - ID (a categorical number for the class)
        - WNID (Wordnet ID)
        - words (description)
        - gloss (long description)
        - num_children
        - children
        - wordnet_height
        - num_train_images
    """
    if data_dir is None:
        data_dir = os.environ['IMAGENET2017_DIR']

    from scipy.io import loadmat
    x = loadmat(os.path.join(data_dir, 'devkit', 'data', 'meta_clsloc.mat'))

    def item_to_dict(item):
        y = {}
        y['ID'] = item[0][0,0]
        y['WNID'] = str(item[1][0])
        y['name'] = str(item[2][0])
        y['description'] = str(item[3][0])
        y['num_train_images'] = item[4][0,0]
        return y

    return [item_to_dict(y) for y in x['synsets'][0]]


def get_validation_labels(data_dir=None, omit_blacklist=True):
    """ Gets the validation labels for the imagenet validation set.
    Returns the results as (filename, label) """
    if data_dir is None:
        data_dir = os.environ['IMAGENET2017_DIR']

    val_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'val')
    files = os.listdir(val_dir)
    files = [os.path.join(val_dir, f) for f in files]
    assert len(files) == 50000
    label_f = 'ILSVRC2015_clsloc_validation_ground_truth.txt'
    black_f = 'ILSVRC2015_clsloc_validation_blacklist.txt'
    with open(os.path.join(data_dir, 'devkit', 'data', label_f), 'r') as f:
        labels = f.readlines()
    labels = [int(l.strip()) for l in labels]

    data = list(zip(files, labels))

    if omit_blacklist:
        with open(os.path.join(data_dir, 'devkit', 'data', black_f), 'r') as f:
            blacklist = f.readlines()
        blacklist = ['ILSVRC2012_val_{:08d}.JPEG'.format(int(l.strip()))
                     for l in blacklist]
        data = [d for d in data if os.path.basename(d[0]) not in blacklist]

    return data


def get_clsloc_queues(base_dir, img_size=None, transform=None,
                      maxsize=1000, num_threads=(2,2,2),
                      max_epochs=float('inf'), get_queues=(True, True, True),
                      _rand_data=False):
    """ Get Image queues for ImageNet

    Parameters
    ----------
    base_dir: str
        Path to the folder containing the ImageNet data. Should be the root of
        the folder, i.e. will have directory strucure::

            base_dir
            |-- Annotations
            |-- Data
            |-- ImageSets

    img_size : tuple(int, int) or None
        What image size to load it in as (in height and width pixels).  This can
        be used alongside the transform parameter to resize the image. If this
        value is not None, the image loader will use Pillow's
        :py:meth:`PIL.Image.resize` to resize it to this shape using BILINEAR
        interpolation. Of course you can also do reshaping in the transform
        callable, if you write it so as to make it accept flexible sized numpy
        arrays. A value of None will keep the image in its loaded size.
   transform : None or callable or tuple(callable, callable, callable)
        Transformation function to apply to images to the train, test and val
        queues (in that order). A single callable will use the same function for
        all 3. The callable(s) should be function(s) that accept a numpy array
        representing **one** image, and transforms it/preprocesses it. E.g. you
        may want to remove the mean and divide by standard deviation before
        putting into the queue.
    maxsize : int or tuple(int, int, int)
        How big the train, test and val queues will be (in that order). A single
        int will use the same value for all 3. Increase this if your main
        program is chewing through the data quickly, but increasing it will also
        mean more memory is taken up.
    num_threads : int or tuple(int, int, int)
        How many threads to use for the train, test and validation threads (in
        that order). A single int will use the same value for all 3.
    max_epochs : int
        How many epochs to run before returning FileQueueDepleted exceptions
    get_queues : tuple(bool, bool, bool)
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
    # Check the data directory has the folders we expect
    required = ['Annotations', 'Data', 'ImageSets']
    actual = os.listdir(base_dir)
    present = [True for i in required if i in actual]
    if False in present:
        raise ValueError(
            "The provided data_dir isn't pointing to the expected position " +
            "in the ImageNet directory. It should be pointing at the folder " +
            "containing the 'Annotations', 'Data' and 'ImageSets' directories")

    # Set some useful directories
    img_dir = os.path.join(base_dir, 'Data', 'CLS-LOC')
    imgset_dir = os.path.join(base_dir, 'ImageSets', 'CLS-LOC')

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

    # Get the synsets - a list of the class info.
    synsets = load_synsets()
    lookup = {k['WNID']: k['ID'] for k in synsets}

    if not _rand_data:
        train_queue = None
        if get_queues[0]:
            # Load the training file list
            cls_train_list = os.path.join(imgset_dir, 'train_cls.txt')
            with open(cls_train_list, 'r') as f:
                x = f.readlines()

            # The data in the file comes in the format:
            #
            #   'n01440764/n01440764_10026 50'
            #
            # Where the first part is the path to the image (dir/filename) and
            # the second part is the image index. We don't need this index so
            # split on the space in the string.
            files = [k.split(' ')[0]+'.JPEG' for k in x]

            # To get the labels, use the synset from the file name. Split on the
            # forward slash to get this
            labels = [lookup[f.split('/')[0]]-1 for f in files]
            labels = utils.convert_to_one_hot(labels, 1000)

            # Create a file queue from this
            file_queue = core.FileQueue()
            file_queue.load_epochs(list(zip(files, labels)),
                                   max_epochs=max_epochs)

            # Create an image queue for this file queue
            train_queue = core.ImgQueue(maxsize=train_qsize,
                                        name='ImageNet Train Queue')
            train_queue.start_loaders(
                file_queue, num_threads=train_threads, img_size=img_size,
                img_dir=os.path.join(img_dir, 'train'), transform=train_xfm)

        test_queue = None
        if get_queues[1]:
            cls_test_list = os.path.join(imgset_dir, 'test.txt')
            with open(cls_test_list, 'r') as f:
                x = f.readlines()
            files = [k.split(' ')[0]+'.JPEG' for k in x]

            # Create a file queue from this
            file_queue = core.FileQueue()
            file_queue.load_epochs(files)

            # Create an image queue for this file queue
            test_queue = core.ImgQueue(maxsize=test_qsize,
                                       name='ImageNet Test Queue')
            test_queue.start_loaders(
                file_queue, num_threads=test_threads, img_size=img_size,
                img_dir=os.path.join(img_dir, 'test'), transform=test_xfm)

        val_queue = None
        if get_queues[2]:
            # Load the validation file list
            cls_val_list = os.path.join(imgset_dir, 'val.txt')
            cls_val_labels = os.path.join(
                os.path.dirname(__file__),
                'ILSVRC2014_clsloc_validation_ground_truth.txt')
            cls_val_blacklist = os.path.join(
                os.path.dirname(__file__),
                'ILSVRC2014_clsloc_validation_blacklist.txt')

            with open(cls_val_list, 'r') as f:
                x = f.readlines()
            with open(cls_val_labels, 'r') as f:
                y = f.readlines()
            with open(cls_val_blacklist, 'r') as f:
                z = f.readlines()

            files = [k.split(' ')[0]+'.JPEG' for k in x]
            labels = [int(y[int(k.split(' ')[1])-1])-1 for k in x]
            labels = utils.convert_to_one_hot(labels, 1000)
            blacklist = [int(k)-1 for k in z]
            whitelist = np.ones((len(files),))
            whitelist[blacklist] = 0

            files = [f for keep, f in zip(whitelist, files) if keep]
            labels = [l for keep, l in zip(whitelist, labels) if keep]

            # Create a file queue from this
            file_queue = core.FileQueue()
            file_queue.load_epochs(list(zip(files, labels)))

            # Create an image queue for this file queue
            val_queue = core.ImgQueue(maxsize=val_qsize,
                                      name='ImageNet Val Queue')
            val_queue.start_loaders(
                file_queue, num_threads=val_threads, img_size=img_size,
                img_dir=os.path.join(img_dir, 'val'), transform=val_xfm)

    else:
        pass
        # Randomly generate some image like data
        #  tr_data = np.random.randint(255, size=(10000, 224, 224, 3))
        #  tr_labels = np.random.randint(10, size=(10000,))
        #  te_data = np.random.randint(255, size=(1000, 32, 32, 3))
        #  te_labels = np.random.randint(10, size=(1000,))
        #  val_data = np.random.randint(255, size=(1000, 32, 32, 3))
        #  val_labels = np.random.randint(10, size=(1000,))
        #  # convert to one hot
        #  tr_labels = utils.convert_to_one_hot(tr_labels)
        #  te_labels = utils.convert_to_one_hot(te_labels)
        #  val_labels = utils.convert_to_one_hot(val_labels)

    # allow for the filling of the queues with some samples
    time.sleep(0.5)
    return train_queue, test_queue, val_queue
