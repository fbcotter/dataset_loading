MNIST
=====

As this is a very commonly used dataset, there is a utility function to help
load it in:

This is identical to the CIFAR function. To check out usage instructions, have
a look `there`__. **Note that the return size for MNIST will be 28x28x1**.

In particular, there exists two functions in this module that may be of use. The
first is :py:func:`dataset_loading.mnist.load_mnist_data`, which can be used to
load in MNIST without queues. There is an argument for this function to request
it to download MNIST if you haven't already got it. 

The second is :py:func:`dataset_loading.cifar.get_mnist_queues` which will load
MNIST and put it into some queues. Although MNIST is very small and can easily
fit into memory, the benefit of this is parallel processing can be used to
prescale the data before feeding it to your network. For more examples on how to
do this, see the page explaining loading in the `CIFAR`__ data.

__ http://dataset-loading.readthedocs.io/en/latest/cifar.html 
__ http://dataset-loading.readthedocs.io/en/latest/cifar.html 
