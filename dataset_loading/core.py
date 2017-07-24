from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import queue
import threading
from random import shuffle
from PIL import Image
import os
import math

EPOCHS_TO_PUT = 10


def catch_empty(func, handle=lambda e: e, *args, **kwargs):
    """ Returns the empty exception rather than raising it

    Useful for calling queue.get in a list comprehension
    """
    try:
        return func(*args, **kwargs)
    except queue.Empty as e:
        return handle(e)


class ImgQueueNotStarted(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class FileQueueNotStarted(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class FileQueueDepleted(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ImgQueue(queue.Queue):
    """A queue to hold images

    This queue can hold images which will be loaded from the main program.
    Multiple file reader threads can fill up this queue as needed to make sure
    demand is met.

    Each entry in the image queue will then be either tuple of (data, label).
    If the data is loaded using a filename queue and image loader threads and a
    label is not provided, each queue item will still be a tuple, only the label
    will be None.

    To get a batch of samples from the ImageQueue, see the :py:meth:`get_batch`
    method.
    """
    def __init__(self, maxsize=1000):
        queue.Queue.__init__(self, maxsize=maxsize)
        self.epoch_size = None
        self.read_count = 0
        self.loaders_started = False
        self.last_batch = False

    def start_loaders(self, file_queue, num=3, img_dir='', img_size=None,
                      transform=None):
        """Starts the threads to load the images into the ImageQueue

        Parameters
        ----------
        file_queue : FileQueue object
            An instance of the file queue
        num : int
            How many parallel threads to start to load the images
        img_dir : str
            Offset to add to the strings fetched from the file queue so that a
            call to load the file in will succeed.
        img_size : tuple of (height, width) or None
            What size to resize all the images to. If None, no resizing will be
            done.
        transform : function handle or None
            Pre-filtering operation to apply to the images before adding to the
            Image Queue. If None, no operation will be applied. Otherwise, has
            to be a function handle that takes the numpy array and returns the
            transformed image as a numpy array.
        """
        self.file_queue = file_queue
        self.epoch_size = file_queue.epoch_size
        loaders = [
            ImgLoader('Loader {}'.format(i+1), file_queue, self,
                      img_dir=img_dir, img_size=img_size, transform=transform)
            for i in range(num)
        ]
        [loader.start() for loader in loaders]
        self.loaders = loaders
        self.loaders_started = True

    def get_batch(self, batch_size=None, block=False, timeout=3):
        """Tries to get a batch from the Queue.

        If there is less than a batch of images, it will grab them all.
        If the epoch size was set and the tracking counter sees there are
        fewer than <batch_size> images until we hit an epoch, then it will
        cap the amount of images grabbed to reach the epoch.

        Parameters
        ----------
        batch_size : int
            How many samples we want to get.
        block : bool
            Whether to block (and wait for the img queue to catch up)
        timeout : bool
            How long to wait on timeout

        Returns
        -------
        out: list of samples
            the batch of items

        Notes
        -----
        When we pull the last batch from the image queue, the flag last_batch
        is set to true. This allows the calling function to synchronize tests
        with the end of an epoch.

        Raises
        ------
        FileQueueNotStarted - when trying to get a batch but the file queue
        manager hasn't started.
        FileQueueDepleted -  when we have hit the epoch limit.
        ImgQueueNotStarted - when trying to get a batch but no image loaders
        have started.
        queue.Empty - If timed out on trying to read an image
        """
        if not self.loaders_started:
            raise ImgQueueNotStarted('''Start the Image Queue Loaders by calling
                start_loaders before calling get_batch''')

        # Determine some limits on how many images to grab.
        rem = batch_size
        if self.epoch_size is not None:
            rem = self.epoch_size - self.read_count

        # Pull some samples from the queue - don't block and if we hit an
        # empty error, just keep going (don't want to block the main loop)
        nsamples = min(rem, batch_size)
        if block:
            data = [self.get(block=True, timeout=timeout) for _ in range(nsamples)]
        else:
            data = [catch_empty(lambda: self.get(block=block))
                    for _ in range(nsamples)]
            data = [x for x in data if type(x) is not queue.Empty]

        if len(data) == 0:
            if not self.file_queue.started:
                raise FileQueueNotStarted('''Start the File Queue manager by calling
                    FileQueue.load_epochs before calling get_batch''')
            elif self.file_queue.started and not self.file_queue.filling and \
                 self.file_queue.qsize() == 0:
                raise FileQueueDepleted('End of Training samples reached')

        if self.epoch_size is not None:
            last_batch = (len(data) + self.read_count) >= self.epoch_size
            if last_batch:
                self.read_count = len(data) + self.read_count - self.epoch_size
                self.last_batch = True
            else:
                self.read_count += len(data)
                self.last_batch = False

        return data


class FileQueue(queue.Queue):
    """A queue to hold filename strings

    This queue is used to indicate what order of jpeg files should be read. It
    may also be a good idea to put the class label alongside the filename as a
    tuple, so the main program can get access to both of these at the same time.

    Create the class, and then call the load_epochs() method to start a thread
    to manage the queue and refill it as it gets low.
    """
    def __init__(self, maxsize=0):
        queue.Queue.__init__(self, maxsize=maxsize)
        self.epoch_count = -1
        self.thread = None
        self.epoch_size = None

        # Flags for the ImgQueue
        self.filling = False
        self.started = False

    def get(self, block=True, timeout=0):
        if not self.started:
            raise FileQueueNotStarted(
                'Call load_epochs before trying to pull from the file queue')
        else:
            return super(FileQueue, self).get(block=block, timeout=timeout)

    def load_epochs(self, files, reshuffle=True, max_epochs=math.inf):
        """
        Starts a thread to load the file names into the file queue.

        Parameters
        ----------
        files : list
            Can either be a list of filename strings or a list of tuples of
            (filenames, labels)
        reshuffle : bool
            Whether to shuffle the list before adding it to the queue
        max_epochs : int or math.inf
            Maximum number of epochs to allow before queue manager stops
            refilling the queue.

        Raises
        ------
        ValueError - If the files queue was empty
        """
        if len(files) == 0:
            raise ValueError('The files list cannot be empty')

        # Limit ourselves to only one thread for the file queue
        if self.thread is None:
            myfiles = files[:]
            self.max_epochs = max_epochs
            self.thread = threading.Thread(
                target=self.manage_queue, name='File Queue Thread',
                kwargs={'files': myfiles, 'reshuffle': reshuffle}, daemon=True)
            self.thread.start()

    def manage_queue(self, files, reshuffle=True):
        self.started = True
        self.filling = True
        self.epoch_count = 0
        self.epoch_size = len(files)

        while self.epoch_count < self.max_epochs:
            if self.qsize() < 0.5*len(files):
                epochs_to_put = min(
                    EPOCHS_TO_PUT, self.max_epochs - self.epoch_count)
                # Load multiple epochs in at a time
                for i in range(epochs_to_put):
                    if reshuffle:
                        shuffle(files)
                    [self.put(item) for item in files]
                    self.epoch_count += 1

        self.filling = False


class ImgLoader(threading.Thread):
    """ A thread to load in images from a filename queue into an image queue.
    """
    def __init__(self, name, file_queue, img_queue, img_size=None,
                 img_dir='', transform=None):
        threading.Thread.__init__(self, daemon=True)
        self.name = name
        self.fqueue = file_queue
        self.iqueue = img_queue
        self.img_size = img_size
        self.base_dir = img_dir
        self.transform = transform

    def load_image(self, im=''):
        """ Load an image in and return it as a numpy array.
        """
        img = Image.open(im)
        if self.img_size is not None:
            img = img.resize(self.img_size)
        # Make sure it is 3 channel
        img = img.convert(mode='RGB')
        img_np = np.array(img).astype(np.float32)
        if self.transform is not None:
            img_np = self.transform(img_np)

        return img_np

    def run(self):
        print("Starting " + self.name)
        if not self.fqueue.started:
            raise FileQueueNotStarted()

        while True:
            # Try get an item - the file queue running out is the main way for
            # this thread to exit.
            try:
                item = self.fqueue.get_nowait()
            except:
                if not self.fqueue.filling:
                    return

            # Split the item into a filename and label
            try:
                f, label = item
            except:
                f = item
                label = None

            img = self.load_image(os.path.join(self.base_dir, f))
            self.iqueue.put((img, label))
            self.fqueue.task_done()


def convert_to_one_hot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)
