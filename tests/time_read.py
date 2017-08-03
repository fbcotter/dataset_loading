from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import timeit
from time import sleep # noqa
from dataset_loading import cifar # noqa
import io
from contextlib import redirect_stdout
import argparse


def test_time(args):
    setup = """
def transform(x):
    return x-np.mean(x)

train_queue, test_queue, val_queue = cifar.get_cifar_queues(
    '', maxsize=1000, transform=transform, _rand_data=True)
sleep(1)
    """
    cmd = "train_queue.get_batch(100, block=True)"""
    f = io.StringIO()
    with redirect_stdout(f):
        a = timeit.repeat(cmd, setup, number=10, globals=globals())

    if args.print:
        print(a)
    else:
        print(np.median(a))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-print', dest='print', action='store_false',
                        default=True)
    args = parser.parse_args()
    test_time(args)
