CIFAR 10 & 100 Datasets
=======================

As this is a very commonly used dataset, there is a utility function to help
load it in:

.. automodule:: dataset_loading.cifar
    :members: get_cifar_queues
    :noindex:

The best way to understand this function is to see how it is used.

.. def get_cifar_queues(data_dir, cifar10=True, val_size=2000, transform=None,
..                   max_qsize=1000, num_threads=(2,2,2),
..                     max_epochs=float('inf'), _rand_data=False):

.. code:: python

    from dataset_loading import cifar
    from time import sleep
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '/path/to/cifar/data', cifar10=True)
    sleep(1)
    data, labels = train_queue.get_batch(100)
    test, labels = test_queue.get_batch(100)
    val, labels = val_queue.get_batch(100)

Preprocessing
-------------
Ok cool, what if we want to preprocess images by removing their mean before
putting them into the queue. The benefit of this is that when your main function
is ready for the next batch, it doesn't have to do any of this preprocessing.

.. code:: python
    
    from dataset_loading import cifar
    import numpy as np
    from time import sleep
    def preprocess(x):
        x = x.astype(np.float32)
        x = x - np.mean(x)
        return x
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '/path/to/cifar/data', transform=preprocess, cifar10=True)
    sleep(1)
    data, labels = train_queue.get_batch(100)
    test, labels = test_queue.get_batch(100)
    val, labels = val_queue.get_batch(100)

Ok, easy enough. What about if we wanted to do some preprocessing to the train
set, but not to the validation and test? This is commonly done to 'augment' your
dataset.

.. code:: python

    from dataset_loading import cifar
    import numpy as np
    from time import sleep
    # this augmentation just adds noise to the train data
    def preprocess(x):
        x = x.astype(np.float32)
        x = x + 10*np.random.rand(32,32,3)
        return x
    transform = (preprocess, None, None)
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '/path/to/cifar/data', transform=transform, cifar10=True)
    sleep(1)
    data, labels = train_queue.get_batch(100)
    test, labels = test_queue.get_batch(100)
    val, labels = val_queue.get_batch(100)

Epoch Management
----------------
One of the main annoyances with tensorflow was the difficulty of swapping
between train and validation sets in the same main function. Say if you wanted
to process one epoch of training data, then run some validation tests before
getting a new epoch of data. You would have to keep track manually of how many
images you'd read as if you tried to set an epoch limit to 1, and then restart
the queues, you would run into all sorts of problems.

The ImgQueue in this package has a `last_batch` property that indicates whether this
epoch was the last one or not, providing an easy indication for the main program
to move onto the validation stage. **This flag will get reset if you read from
it**.

.. code:: python

    from dataset_loading import cifar
    import numpy as np
    from time import sleep
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '/path/to/cifar/data', cifar10=True)
    sleep(1)
    while True:
        while not train_queue.last_batch:
            data, labels = train_queue.get_batch(100)
            # process the data

        # Do some validation testing then
        # loop back to beginning and get the next batch

You can also inspect how many images have been processed in the current epoch by
looking at the ImgQueue.read_count property. This shouldn't be modified however,
as then the file queues and the image queue will get out of sync.

You can put a limit on the epoch count too. When this limit is reached,
a `dataset_loading.FileQueueDepleted` exception will be raised:

.. code:: python

    from dataset_loading import cifar, FileQueueDepleted
    import numpy as np
    from time import sleep
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '/path/to/cifar/data', cifar10=True, max_epochs=50)
    try: 
        while not train_queue.last_batch:
            data, labels = train_queue.get_batch(100)
            # process the data

        # Do some validation testing then
        # loop back to beginning and get the next batch
    except FileQueueDepleted:
        # No need to do any join calls for the threads as these should already
        # have exited, and if they haven't, they're daemon threads so no
        # worries.
        print('All done')

Selecting Queues
----------------
If you only want to get the train queue or the train and validation queues say,
you can do this by using the `get_queues` parameter. E.g.:

.. code:: python

    from dataset_loading import cifar, FileQueueDepleted
    import numpy as np
    from time import sleep
    train_queue, test_queue, val_queue = cifar.get_cifar_queues(
        '/path/to/cifar/data', cifar10=True, get_queues=(True, False, True))
    assert test_queue is None
