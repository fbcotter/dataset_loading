Dataset Loading
===============

This repo is aimed at being a centralized resource for loading in commonly used
image datasets like CIFAR, PASCAL VOC, MNIST, ImageNet and others.

Some of these datasets will fit easily on disk (CIFAR and MNIST), but many of
the others won't. This means we have to set up threads to load them as we need
them into memory. Tensorflow provides some ability to do this, but after
several attempts at using these resources, I found it far too opaque and
difficult to use. This package does essentially the same thing as what
tensorflow does, but using python's threading and queue modules.

Usage
-----
For the bigger datasets, we need 2 queues and several threads to load images in
parallel.

Firstly, a FileQueue_ is used to store a list of file names (e.g.
jpegs).  This is also the location of sequencing (there is an option to shuffle
the entries in this queue when adding) and where we set the limits on the
number of epochs processed (if we wish to). For example, this would set up
a file queue for 50 epochs:: 

    import dataset_loading as dl
    files = os.listdir(<path_to_images>)
    files = [f for f in files if os.splitext(f)[1] == '.jpeg']
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files, max_epochs=50)

.. _FileQueue: http://dataset-loading.readthedocs.io/en/latest/filequeue.html#filequeue

Next we create an ImageQueue_ to hold a set amount of images (not
the entire batch, but enough to keep the main program happily fed). This class has
a method we call for starting image reader threads (again, you can choose how
many of these you need to meet your main's demand). Following the above code,
you could add an image queue like so::

    img_queue = dl.ImgQueue()
    img_queue.start_loaders(file_queue, num=3)

.. _ImageQueue: http://dataset-loading.readthedocs.io/en/latest/imagequeue.html#imagequeue

In the main function, we call the ImageQueue's
`ImgQueue.get_batch`__ 
to get a batch of images from the ImageQueue::

    # Wait for the image queue to fill up
    sleep(5)
    img_queue.get_batch(<batch_size>)

__ http://dataset-loading.readthedocs.io/en/latest/functions.html#dataset_loading.core.ImgQueue.get_batch

For synchronization with epochs, the ImageQueue has an attribute `last_batch`
that will be set to true when an epoch's worth of images have been pulled from
the ImageQueue. 

If you want to pre-process images before putting them into the image queue, you
can provide a callable function to `ImgQueue.start_loaders` to do this (see its 
docstring for more info).

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

I would recommend to download and install (with the editable flag), as it is
likely you'll want to tweak things/add functions more quickly than I can handle
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
