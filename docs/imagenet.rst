ImageNet (2017)
===============

As this is a very commonly used dataset, there are some utility functions to
load it in. This can be very lengthy to set up, particularly as the ImageNet
mirror is slow these days. You will need an account with ImageNet first to get
access to the tar files. Once you have this, visit the `data archive`__ for
ImageNet and go to the 2017 download page. 

Downloading
-----------

Download the following 4 tar files. I used wget, but the more advanced of you
may want to use a download manager. Note that to do this, you'll need to your
username and access key (attainable from your account page in ImageNet):

.. code:: bash

    cd /path/to/ImageNet2017/raw
    wget http://image-net.org/image/ILSVRC2017/ILSVRC2017_devkit.tar.gz
    wget http://image-net.org/image/ILSVRC2017/ILSVRC2017_CLS-LOC.tar.gz?username=[username]&accesskey=[accesskey]
    wget http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET.tar.gz?username=[username]&accesskey=[accesskey] 
    wget http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET_test_new.tar.gz?username=[username]&accesskey=[accesskey]

This can several days due to the size of these files (the CLS-LOC file is
155GB).

Preparation
-----------
Once you've downloaded the giant files, take make sure you run md5sum to make
sure that your files match up. The md5s you should get are:

- CLS-LOC dataset: 099d21920ef427c1bedc0d5d182277cf
- DET dataset: 237b95a860e9637b6a27683268cb305a
- DET test dataset: e9c3df2aa1920749a7ec35d1847280c6

Now you can extract the files. This will also take quite some time (several
days). The commands are:

.. code:: bash
    
    tar xzvf ILSVRC2017_CLS-LOC.tar.gz -C /path/to/ImageNet2017
    tar xzvf ILSVRC2017_CLS-LOC.tar.gz -C /path/to/ImageNet2017
    tar xzvf ILSVRC2017_CLS-LOC.tar.gz -C /path/to/ImageNet2017

After doing this, running `tree -L 3` from the ImageNet2017 base folder should give 
you the following output::

    .
    ├── Annotations
    │   └── CLS-LOC
    │       ├── train
    │       └── val
    ├── Data
    │   └── CLS-LOC
    │       ├── test
    │       ├── train
    │       └── val
    ├── devkit
    │   ├── COPYING
    │   ├── data
    │   │   ├─ ...
    │   ├── evaluation
    │   │   ├─ ...
    │   └── readme.txt
    └── ImageSets
        └── CLS-LOC
            ├── test.txt
            ├── train_cls.txt
            ├── train_loc.txt
            └── val.txt


Sample
------
