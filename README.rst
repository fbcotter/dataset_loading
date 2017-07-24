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

Firstly, a :ref:`FileQueue-label` is used to store a list of file names (e.g.
jpegs).  This is also the location of sequencing (there is an option to shuffle
the entries in this queue when adding) and where we set the limits on the
number of epochs processed (if we wish to). 

Next we create an :ref:`ImageQueue-label` to hold a set amount of images (not
the entire batch, but enough to not hold up the main program). This class has
a method we call for starting image reader threads (again, you can choose how
many of these you need to meet your main's demand).

In the main function, we call the ImageQueue's
:py:meth:`ImgQueue.get_batch <dataset_loading.core.ImgQueue.get_batch>` 
to get a batch of images from the ImageQueue. For synchronization with epochs,
the ImageQueue has an attribute `last_batch` that will be set to true when an
epoch's worth of images have been pulled from the ImageQueue. See the docstring
of
:py:class:`ImgQueue <dataset_loading.core.ImgQueue>` for more information.

Installation
------------
Direct install from github (useful if you use pip freeze). To get the master
branch, try::

    $ pip install -e git+https://github.com/fbcotter/dataset_laoding#egg=dataset_loading

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

There is `more documentation <http://dataset-loading.readthedocs.io>`_
available online and you can build your own copy via the Sphinx documentation
system::

    $ python setup.py build_sphinx

Compiled documentation may be found in ``build/docs/html/`` (index.html will be
the homepage)
