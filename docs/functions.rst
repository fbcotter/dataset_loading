API Guide
=========

Core Functions
--------------

.. automodule:: dataset_loading
    :members: FileQueue, ImgQueue
    :show-inheritance:

Exceptions
----------

.. autoexception:: dataset_loading.ImgQueueNotStarted
.. autoexception:: dataset_loading.FileQueueNotStarted
.. autoexception:: dataset_loading.FileQueueDepleted

Dataset Specific
----------------

MNIST
~~~~~
.. automodule:: dataset_loading.mnist
    :members: 
    :show-inheritance:

CIFAR
~~~~~
.. automodule:: dataset_loading.cifar
    :members: load_cifar_data, get_cifar_queues
    :show-inheritance:

PASCAL
~~~~~~
.. automodule:: dataset_loading.pascal
    :members: 
    :show-inheritance:

