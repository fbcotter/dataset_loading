Dataset Loading
===============

This repo is aimed at being a centralized resource for loading in commonly used
image datasets like CIFAR, PASCAL VOC, MNIST, ImageNet and others.

Some of these datasets will fit easily on disk (CIFAR and MNIST), but many of
the others won't. This means we have to set up threads to load them as we need
them into memory. Tensorflow provides some ability to do this, but after
several attempts at using these resources, we found them far too opaque and
difficult to use. This package does essentially the same thing as what
tensorflow does, but using python's threading, multiprocessing and queue
packages. 



Threads vs Processes
--------------------
Initially this package would only use Python's threading package to parallelize
tasks. It quickly became apparent that this caps the benefits of
parallelization, as all of these threads will only take up to 1 processor core.
In reality, we want to be able to take up more processors for data loading to
reduce bottlenecks. It is still untested, but we are adding in multiprocess
support for the heavy lifting tasks (in particular, loading and preprocessing
images into `The ImageQueue`_).

Dataset Specific Usage
----------------------
For instructions on how to call the functions to get images in for common
datasets, see their help pages. These functions wrap around the `General Usage`_
functions and are provided for convenience. If your application doesn't quite
fit into these functions, or if you have a new dataset, have a look at `General
Usage`_, as it was designed to make queueing for any dataset type as easy as
possible.

- `MNIST usage instructions`__
- `CIFAR10/CIFAR100 usage instructions`__

__ http://dataset-loading.readthedocs.io/en/latest/mnist.html 
__ http://dataset-loading.readthedocs.io/en/latest/cifar.html 

General Usage
-------------
For the bigger datasets, we need 2 queues and several threads (perhaps on
multiple processors) to load images in parallel.

1. A File Queue to store a list of file names.
   Sequencing can be done by shuffling the file names before inserting into the
   queue. 

   - One thread should be enough to manage this queue.

2. An Image Queue to load images into.

   - Several threads will likely be needed to read file names from the file
     queue, load from disk, and put into the Image Queue. We may get away with
     running these all in one Python process, but may need to use more.


The FileQueue
~~~~~~~~~~~~~
A FileQueue_ is used to store a list of file names (e.g.  jpegs).  This is also
the location of sequencing (there is an option to shuffle the entries in this
queue when adding) and where we set the limits on the number of epochs processed
(if we wish to). For example, this would set up a file queue for 50 epochs: 

.. code:: python

    import dataset_loading as dl
    IM_DIR = /path/to/images
    files = os.listdir(IM_DIR)
    files = [f for f in files if os.path.splitext(f)[1] == '.jpeg']
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files, max_epochs=50)
    ...
    ...
    file_queue.join_loaders()

The `load_epochs` method will also start a single thread to manage the queue and
refill it if it's getting low (shuffling along as it goes).

If you know what the labels are, you should also feed them to the File Queue
alongside the file names in a list of (file, label) tuples. E.g.:

.. code:: python

    # Assume <labels> is a list of all of the labels and <files> is a 
    # list of the files.
    file_queue = dl.FileQueue()
    file_queue.load_epochs(list(zip(files, labels)), max_epochs=float('inf'))

Note that when you are done with the queue, you should call the queue's
`join_loaders` method, which will make sure the queue is empty and the loader
thread exits.

The ImageQueue
~~~~~~~~~~~~~~
An ImageQueue_ to hold a set amount of images (not the entire batch, but enough
to keep the main program happily fed). This class has a method we call for
starting image reader threads (again, you can choose how many of these you need
to meet your main's demand). Following the above code, you add an image
queue like so:

.. code:: python

    img_queue = dl.ImgQueue(maxsize=1000)
    img_queue.start_loaders(file_queue, num_threads=3, img_dir=IM_DIR)
    # Wait for the image queue to fill up
    sleep(2)
    data, labels = img_queue.get_batch(batch_size=100)
    ...
    ...
    img_queue.join_loaders()

The ImgQueue.start_loaders_ method will start `num_threads` threads, each of
which read from the file_queue, load from disk and feed into the image queue.

If you want the loaders to pre-process images before putting them into the image
queue, you can provide a callable to ImgQueue.start_loaders_ to do this (see its
docstring for more info). For example:

.. code:: python

    img_queue = dl.ImgQueue()
    def preprocess(x):
        x = x.astype(np.float32)
        x = x - np.mean(x)
        x = x/max(1, np.std(x))
        return x
    img_queue.start_loaders(file_queue, num_threads=3, transform=preprocess)

The ImgQueue.get_batch_ method has two extra options (`block` and `timeout`),
instructing it how to handle cases when the image queue doesn't have a full
batch worth of images (should we return with whatever's there, or wait for the
loaders to catch up?). See its docstring for more info.

For synchronization with epochs, the ImageQueue has an attribute `last_batch`
that will be set to true when an epoch's worth of images have been pulled from
the ImageQueue. 

.. code:: python

    data, labels = img_queue.get_batch(batch_size=100)
    last_batch = img_queue.last_batch
    if last_batch:
        # Print summary info...
        
You can monitor the queue size and fetch times for the ImgQueue too (to check
whether you need to tweak some settings). This works by printing out info to
a tensorboard summary file (currently only supported way of doing it). 
All you need to do is create a `tf.summary.FileWriter` (you can use the same one
the rest of your main program is using), and call the ImgQueue.add_logging_
method. This will add the data as a to your tensorboard file.

.. code:: python
    
    img_queue = dl.ImgQueue()
    def preprocess(x):
        x = x.astype(np.float32)
        x = x - np.mean(x)
        x = x/max(1, np.std(x))
        return x
    img_queue.start_loaders(file_queue, num_threads=3, transform=preprocess)
    file_writer = tf.summary.FileWriter('./log', tf.get_default_graph())
    # Write period is the sample period in numbers of batches for dumping data
    img_queue.add_logging(file_writer, write_period=10)

Note that when you are done with the queue, you should call the queue's
`join_loaders` method, which will make sure the queue is empty and the loader
thread exits.

Small Datasets
~~~~~~~~~~~~~~
If you have a special case where the dataset is small, and so can fit into
memory (like CIFAR or MNIST), then you won't need the same complexity to get
batches of data and labels. However, it may still be beneficial to use the
ImgQueue class for two reasons:

- Keeps the same programmatic interface regardless of the dataset
- May still want to parallelize things if you want to do preprocessing of images
  before putting them in the queue.

For this, use ImgQueue.take_dataset_ instead of ImgQueue.start_loaders_.
This method also has options like whether to shuffle the samples or not (will
shuffle by default), and can take a callable function to apply to the images
before putting them in the queue. The default number of threads to create is 1,
but this can be increased with the `num_threads` parameter.

Note: **to avoid duplicating things in memory, the ImgQueue will not copy the
data/labels**. This means that once your main program calls the `take_dataset`
method, it shouldn't modify the arrays.

E.g.

.. code:: python

    import dataset_loading as dl
    import dataset_loading.cifar as dlcifar
    train_d, train_l, test_d, test_l, val_d, val_l = \
        dlcifar.load_cifar_data('/path/to/data')
    img_queue = dl.ImgQueue()
    img_queue.take_dataset(train_d, train_l)
    data, labels = img_queue.get_batch(100)
    # Or say we want to use more parallel threads and morph the image
    def preprocess(x):
        x = x.astype(np.float32)
        x = x - np.mean(x)
        x = x/max(1, np.std(x))
        return x
    img_queue = dl.ImgQueue()
    img_queue.take_dataset(train_d, train_l, num_threads=3, 
                           transform=preprocess)
    data, labels = img_queue.get_batch(100)
     

Installation
------------
Direct install from github (useful if you use pip freeze). To get the master
branch, try::

    $ pip install -e git+https://github.com/fbcotter/dataset_loading#egg=dataset_loading

or for a specific tag (e.g. 0.0.1), try::

    $ pip install -e git+https://github.com/fbcotter/dataset_loading.git@0.0.1#egg=dataset_loading

Download and pip install from Git::

    $ git clone https://github.com/fbcotter/dataset_loading
    $ cd dataset_loading
    $ pip install -r requirements.txt
    $ pip install -e .

It is recommended to download and install (with the editable flag), as it is
likely you'll want to tweak things/add functions more quickly than we can handle
pull requests.

Further documentation
---------------------

There is `more documentation`__
available online and you can build your own copy via the Sphinx documentation
system::

    $ python setup.py build_sphinx

Compiled documentation may be found in ``build/docs/html/`` (index.html will be
the homepage)

__ http://dataset-loading.readthedocs.io
.. _FileQueue: http://dataset-loading.readthedocs.io/en/latest/filequeue.html#filequeue
.. _ImageQueue: http://dataset-loading.readthedocs.io/en/latest/imagequeue.html#imagequeue
.. _ImgQueue.get_batch: http://dataset-loading.readthedocs.io/en/latest/functions.html#dataset_loading.core.ImgQueue.get_batch
.. _ImgQueue.start_loaders: http://dataset-loading.readthedocs.io/en/latest/functions.html#dataset_loading.core.ImgQueue.start_loaders
.. _ImgQueue.take_dataset: http://dataset-loading.readthedocs.io/en/latest/functions.html#dataset_loading.core.ImgQueue.take_dataset
.. _ImgQueue.add_logging: http://dataset-loading.readthedocs.io/en/latest/functions.html#dataset_loading.core.ImgQueue.add_logging
