from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import sleep
from dataset_loading import cifar, FileQueueDepleted
import pytest


def test_getqueue():
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', _rand_data=True)


def test_pullfromqueue():
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', _rand_data=True)
    sleep(1)
    data, labels = train_queue.get_batch(100)
    test, labels = test_queue.get_batch(100)
    val, labels = val_queue.get_batch(100)


def test_transform():
    def transform(x):
        return x-np.mean(x)

    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', transform=transform, _rand_data=True)
    sleep(1)
    data, labels = train_queue.get_batch(100)
    for im in data:
        assert abs(np.mean(im)) <= 0.0001


def test_transform2():
    def transform_tr(x):
        return x-np.mean(x)
    def transform_te(x):
        return x-np.mean(x) + 100

    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', transform=(transform_tr, transform_te, None), _rand_data=True)
    sleep(1)
    data, labels = train_queue.get_batch(100)
    for im in data:
        assert abs(np.mean(im)) <= 0.0001
    data, labels = test_queue.get_batch(100)
    for im in data:
        assert abs(np.mean(im)-100) <= 0.0001


def test_queuemask():
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', get_queues=(True, False, False), val_size=0, _rand_data=True)
    assert test_queue is None
    assert val_queue is None
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', get_queues=(True, False, True), _rand_data=True)
    assert test_queue is None
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', get_queues=(False, True, False), val_size=0, _rand_data=True)
    assert train_queue is None
    assert val_queue is None


def test_lastbatch():
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', _rand_data=True)
    sleep(1)
    train_queue._read_count = train_queue.epoch_size - 200
    data, labels = train_queue.get_batch(100)
    assert not train_queue.last_batch
    data, labels = train_queue.get_batch(100)
    assert train_queue.last_batch


def test_lastbatch_reset():
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', _rand_data=True)
    sleep(1)
    train_queue._read_count = train_queue.epoch_size - 200

    for i in range(2):
        while not train_queue.last_batch:
            data, labels = train_queue.get_batch(100)
        if i == 0:
            old_data = data
        train_queue._read_count = train_queue.epoch_size - 100

    # test that the last_batch flag was reset properly so that the data should
    # be different now
    old_data = np.array(old_data)
    data = np.array(data)
    with pytest.raises(AssertionError, message="Expecting Different arrays"):
        np.testing.assert_array_equal(old_data, data)


def test_epochlimit():
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '', max_epochs=1, _rand_data=True)
    sleep(1)
    epoch_size = train_queue.epoch_size
    for i in range(int(np.ceil(epoch_size//100))):
        data, labels = train_queue.get_batch(100, block=True)

    with pytest.raises(FileQueueDepleted, message="Expecting queue empty exception at epoch limit"):  # noqa
        data, labels = train_queue.get_batch(100)
