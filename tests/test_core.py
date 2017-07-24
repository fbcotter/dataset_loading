from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import queue
import threading
from random import shuffle
from PIL import Image
import os
from time import sleep
import math
import dataset_loading as dl
import pytest


MAX_TIMEOUT = 0.01 # 50 ms
TEST_BASE = os.path.dirname(os.path.realpath(__file__))
PACKAGE_BASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
IMG_DIR = os.path.join(TEST_BASE, 'samples')
files = []


def setup():
    global files
    files = os.listdir(os.path.join(TEST_BASE, 'samples'))
    files = [f for f in files if os.path.splitext(f)[1] == '.jpeg']


def test_filequeue():
    # Test we can create a queue and read from it.
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files)
    for i in range(1000):
        file_queue.get(timeout=MAX_TIMEOUT)


def test_filequeue_max_epochs():
    # Test the max_epoch argument works as expected
    file_queue = dl.FileQueue()

    epoch_size = len(files)
    file_queue.load_epochs(files, max_epochs=2)
    # Eat up everything
    for i in range(2*epoch_size):
        file_queue.get(timeout=MAX_TIMEOUT)
    # Make sure the queue is now empty
    with pytest.raises(queue.Empty):
        file_queue.get(timeout=MAX_TIMEOUT)


def test_filequeue_emptyfiles():
    # Test handling an empty file queue
    files = []
    file_queue = dl.FileQueue()
    with pytest.raises(ValueError):
        file_queue.load_epochs(files)


def test_filequeue_nostart():
    # Test handling an empty file queue
    file_queue = dl.FileQueue()
    with pytest.raises(dl.FileQueueNotStarted):
        file_queue.get()


def test_filequeue_unique():
    # Test that each file in an epoch is unique
    file_queue = dl.FileQueue()

    file_queue.load_epochs(files, max_epochs=1)
    new_files = [file_queue.get(timeout=MAX_TIMEOUT)
                 for _ in range(file_queue.epoch_size)]
    import collections
    assert collections.Counter(new_files) == collections.Counter(files)


def test_filequeue_shuffle():
    # Test the shuffling is working
    file_queue = dl.FileQueue()

    file_queue.load_epochs(files, max_epochs=1)
    new_files = [file_queue.get(timeout=MAX_TIMEOUT)
                 for _ in range(file_queue.epoch_size)]

    same = True
    assert len(files) == len(new_files)
    for f1, f2 in zip(files, new_files):
        if f1 != f2:
            same = False
            break
    assert not same


def test_filequeue_noshuffle():
    # Test the shuffling is working
    file_queue = dl.FileQueue()

    file_queue.load_epochs(files, max_epochs=1, reshuffle=False)
    new_files = [file_queue.get(timeout=MAX_TIMEOUT)
                 for _ in range(file_queue.epoch_size)]

    same = True
    assert len(files) == len(new_files)
    for f1, f2 in zip(files, new_files):
        if f1 != f2:
            same = False
            break
    assert same


def test_imgqueue():
    # Test the imgqueue is working
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files, max_epochs=math.inf)

    img_queue = dl.ImgQueue()
    img_queue.start_loaders(file_queue, num=3, img_dir=IMG_DIR)
    # Allow time to load in data
    sleep(1)
    img_queue.get_batch(100)


def test_imgqueue_epochreached():
    # Test we get the right exception when we've hit the sample limit
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files, max_epochs=1)

    img_queue = dl.ImgQueue()
    img_queue.start_loaders(file_queue, num=3, img_dir=IMG_DIR)
    img_queue.get_batch(batch_size=len(files), block=True)
    with pytest.raises(dl.FileQueueDepleted):
        img_queue.get_batch(1)


def test_imgqueue_nostart():
    # Test we get the right exception when we try to read from the image queue
    # without starting the loaders
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files, max_epochs=1)

    img_queue = dl.ImgQueue()
    with pytest.raises(dl.ImgQueueNotStarted):
        img_queue.get_batch(1)


def test_imgqueue_lots():
    # Test we get the right behaviour when we try to read lots from the image
    # queue without letting it fill up
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files)

    img_queue = dl.ImgQueue()
    img_queue.start_loaders(file_queue, num=3, img_dir=IMG_DIR)

    img_queue.get_batch(1000, block=True)
    img_queue.get_batch(1000, block=True)


def test_lastbatch():
    # Test the status of the last_batch flag is working as expected
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files, max_epochs=10)

    img_queue = dl.ImgQueue()
    img_queue.start_loaders(file_queue, num=3, img_dir=IMG_DIR)

    # Get an entire epoch first
    data = img_queue.get_batch(len(files), block=True)
    assert img_queue.last_batch

    # Get batches of 10 images and test the flag works
    num_batches = np.ceil(len(files)/10).astype('int')
    for b in range(num_batches-1):
        data = img_queue.get_batch(10, block=True)
        assert not img_queue.last_batch
    data = img_queue.get_batch(10, block=True)
    assert img_queue.last_batch

    # Get batches of 13 images and test the flag works
    num_batches = np.ceil(len(files)/13).astype('int')
    for b in range(num_batches-1):
        data = img_queue.get_batch(13, block=True)
        assert not img_queue.last_batch
    data = img_queue.get_batch(13, block=True)
    assert img_queue.last_batch


def test_transform():
    # Test we get the right behaviour when we try to read lots from the image
    # queue without letting it fill up
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files)

    img_queue = dl.ImgQueue()
    def transform(x):
        return x-np.mean(x)

    img_queue.start_loaders(file_queue, num=3, img_dir=IMG_DIR,
                            transform=transform)

    data = img_queue.get_batch(10, block=True)
    for im, _ in data:
        assert abs(np.mean(im)) <= 0.0001
