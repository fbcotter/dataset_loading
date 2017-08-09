from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import queue
import threading
import random
from PIL import Image
import time
import os
import warnings

__all__ = ['FileQueue', 'FileQueueNotStarted', 'FileQueueDepleted',
           'ImgQueue', 'ImgLoader', 'ImgQueueNotStarted']
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
    will be None. If you don't want to return this label, then you can set the
    nolabel input to the start_loaders function.

    To get a batch of samples from the ImageQueue, see the :py:meth:`get_batch`
    method.

    If you are lucky enough to have an entire dataset that fits easily into
    memory, you won't need to use multiple threads to start loading data. You
    may however want to keep the same interface. In this case, you can call the
    take_dataset function with the dataset and labels, and then call the
    :py:meth:`get_batch` method in the same manner.

    Parameters
    ----------
    maxsize : positive int
        Maximum number of images to hold in the queue. Needs to not be 0 or else
        it will keep filling up until you run out of memory.
    name : str
        Queue name

    Raises
    ------
    ValueError if the maxsize parameter is incorrect.
    """
    def __init__(self, maxsize=1000, name=''):
        if maxsize <= 0:
            raise ValueError('The Image Queue needs to have a maximum ' +
                             'size or you may run out of memory.')

        queue.Queue.__init__(self, maxsize=maxsize)

        self._epoch_size = None
        self._read_count = 0
        self.loaders_started = False
        self._last_batch = False
        self.in_memory = False
        self.name = name
        self.logging_on = False
        self._kill = False
        self.file_queue = None

    def __repr__(self):
        def bool2str(x):
            if x:
                return "yes"
            else:
                return "no"
        return ("ImgQueue instance - {}.\n".format(self.name) +
                "Loaders started: {}\n".format(bool2str(self.loaders_started)) +
                "Dataset in mem: {}\n".format(bool2str(self.in_memory)) +
                "Read count: {}\n".format(self.read_count) +
                "Epoch size: {}\n".format(self.epoch_size))

    @property
    def last_batch(self):
        """ Check whether the previously read batch was the last batch in the
        epoch.

        **Reading this value will set it to False**. This allows you to do
        something like this::

            while True:
                while not train_queue.last_batch:
                    data, labels = train_queue.get_batch(batch_size)

                ...
        """
        test = self._last_batch
        if test:
            self._last_batch = False
        return test

    @property
    def epoch_size(self):
        """ The epoch size (as interpreted from the File Queue)
        """
        return self._epoch_size

    @property
    def read_count(self):
        """ Returns how many images have been read in the current epoch.
        """
        return self._read_count

    @property
    def img_shape(self):
        """ Return what the image size is of the images in the queue

        This may be useful to check the output shape after any preprocessing has
        been done.

        Returns
        -------
        img_size : list of ints or None
           Returns the shape of the images in the queue or None if it could not
           determine what they were.
        """
        try:
            return self.queue[0][0].shape
        except:
            return None

    @property
    def label_shape(self):
        """ Return what the label shape is of the labels in the queue

        This may be useful to check the output shape after any preprocessing has
        been done.

        Returns
        -------
        label_shape : list of ints or None
           Returns the shape of the images in the queue or None if it could not
           determine what they were.
        """
        try:
            return self.queue[0][1].shape
        except:
            return None

    def kill_loaders(self):
        """ Method to signal any threads that are filling this queue to stop.

        Threads will clean themselves up if the epoch limit is reached, but in
        case you want to kill them manually before that, you can signal them to
        stop here. Note that if these threads are blocked waiting on input, they
        will still stay alive (and blocked) until whatever is blocking them
        frees up. This shouldn't be a problem though, as they will not be taking
        up any processing power.

        If there is a file queue associated with this image queue, those threads
        will be stopped too.
        """
        self._kill = True
        if self.file_queue is not None:
            self.file_queue.kill_loaders()

    def take_dataset(self, data, labels=None, shuffle=True, num_threads=1,
                     transform=None, max_epochs=float('inf')):
        """Save the image dataset to the class for feeding back later.

        If we don't need a file queue (we have all the dataset in memory), we
        can give it to the ImgQueue class with this method. Images will still
        flow through the queue (so you still need to be careful about how big to
        set the queue's maxsize), but now the preprocessed images will be fed
        into the queue, ready to retrieve quickly by the main program.

        Parameters
        ----------
        data : ndarray of floats
            The images. Should be in the form your main program is happy to
            receive them in, as no reshaping will be done. For example, if the
            data is of shape [10000, 32, 32, 3], then we randomly sample from
            the zeroth axis when we call get batch.
        labels : ndarray numeric or None
            The labels. If not None, the zeroth axis has to match the size of
            the data array. If None, then no labels will be returned when
            calling get batch.
        shuffle : bool
            Normally the ordering will be done in the file queue, as we are
            skipping this, the ordering has to be done here. Set this to true if
            you want to receive samples randomly from data.
        num_threads : int
            How many threads to start to fill up the image queue with the
            preprocessed data.
        transform : None or callable
            Transform to apply to images. Should accept a single image (although
            isn't fussy about what size/shape it is in), and return a single
            image. This will be applied to all the images independently before
            putting them in the Image Queue.

        Notes
        -----
        Even if shuffle input is set to false, that doesn't necessarily mean
        that all images in the image queue will be in the same order across
        epochs. For example, if thread A pulls the first 100 images from the
        list and then thread B gets the second 100. Thread A takes slightly
        longer to process the images than thread B, so these get inserted into
        the Image Queue afterwards.  Trying to synchronize across both queues
        could be done, but it would add unnecessary complications and overhead.

        Raises
        ------
        AssertionError if data and labels don't match up in size.
        """
        assert data.shape[0] == labels.shape[0]
        self._epoch_size = data.shape[0]
        self.data = data
        self.labels = labels
        self.transform = transform

        # Create a file queue. This will only contain indices into the numpy
        # arrays data and labels.
        self.file_queue = FileQueue()
        files = list(range(self._epoch_size))
        self.file_queue.load_epochs(files, shuffle=shuffle,
                                    max_epochs=max_epochs)

        for i in range(num_threads):
            thread = threading.Thread(
                target=self._mini_loaders, name='Mini Loader Thread',
                kwargs={'idx': i+1}, daemon=True)
            thread.start()
        self.loaders_started = True

    def add_logging(self, writer, write_period=10):
        """ Adds ability to monitor queue sizes and fetch times.

        Will try to import tensorflow and throw a warnings.warn if it couldn't.

        Parameters
        ----------
        file_writer : tensorflow FileWriter object
            Uses this object to write out summaries.
        write_period : int
            After how many calls to get_batch should we write to the logger.
        """
        try:
            from dataset_loading.tensorboard_logging import Logger
        except ImportError:
            warnings.warn('Sorry, I couldnt import the necessary modules')
            return
        self.logger = Logger(writer=writer)
        self.write_period = write_period
        self.call_count = 0
        self.logging_on = True
        self.logger_info = {
            'call_count': 0,
            'av_qsize': 0,
            'av_fetch_time': 0,
            'epoch_idx': 0,
            'epoch_size': 0,
            'epoch_qsize': np.zeros((10000,)),
            'epoch_fetch_time': np.zeros((10000,))
        }

    def _mini_loaders(self, idx):
        """ Queue manager for when we have a dataset provided

        This will spin up a thread to load images from the self.data array,
        preprocess them, and put them in the Image Queue.
        """
        print("Starting processing thread {} for {}".format(idx, self.name))
        if not self.file_queue.started:
            raise FileQueueNotStarted(
                "File Queue has to be started before reading from it")

        while not self._kill:
            # Try get an item
            try:
                #  item = self.file_queue.get_nowait()
                item = self.file_queue.get()
                # Split the item into a filename and label
                try:
                    f, label = item
                except:
                    f = item
                    label = None

                # 'Load' the image and label - reshape if necessary
                img = self.data[f]
                if self.transform is not None:
                    img = self.transform(img)
                label = self.labels[f]

                # Put it into my queue.
                self.put((img, label))

                self.file_queue.task_done()
            except queue.Empty:
                # If the file queue ran out, exit quietly
                if not self.file_queue.filling:
                    return

    def start_loaders(self, file_queue, num_threads=3, img_dir='',
                      img_size=None, transform=None, nolabel=False):
        """Starts the threads to load the images into the ImageQueue

        Parameters
        ----------
        file_queue : FileQueue object
            An instance of the file queue
        num_threads : int
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

        Raises
        ------
        ValueError: if called after take_dataset.
        """
        if self.in_memory:
            raise ValueError(
                "You have already called take_dataset for this Image Queue, " +
                "which loaded the images into memory. You cannot start " +
                "threads to load from a file queue afterwards.")
        self.file_queue = file_queue
        self._epoch_size = file_queue.epoch_size
        loaders = [
            ImgLoader('Loader {}'.format(i+1), file_queue, self,
                      img_dir=img_dir, img_size=img_size, transform=transform)
            for i in range(num_threads)
        ]
        [loader.start() for loader in loaders]
        self.loaders = loaders
        self.loaders_started = True

    def get_batch(self, batch_size, block=False, timeout=3):
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
        data : list of ndarray
            List of numpy arrays representing the transformed images.
        labels : list of ndarray or None
            List of labels. Will be None if there were no labels in the
            FileQueue.

        Notes
        -----
        When we pull the last batch from the image queue, the property
        last_batch is set to true. This allows the calling function to
        synchronize tests with the end of an epoch.

        Raises
        ------
        FileQueueNotStarted - when trying to get a batch but the file queue
        manager hasn't started.
        FileQueueDepleted -  when we have hit the epoch limit.
        ImgQueueNotStarted - when trying to get a batch but no image loaders
        have started.
        queue.Empty - If timed out on trying to read an image
        """
        # The data is being fed by queueing threads.
        if not self.loaders_started:
            raise ImgQueueNotStarted(
                "Start the Image Queue Loaders by calling start_loaders " +
                "before calling get_batch")

        # Determine some limits on how many images to grab.
        rem = batch_size
        if self.epoch_size is not None:
            rem = self.epoch_size - self._read_count

        # Pull some samples from the queue - don't block and if we hit an
        # empty error, just keep going (don't want to block the main loop)
        nsamples = min(rem, batch_size)
        start = time.time()
        if block:
            try:
                data = [self.get(block=True, timeout=timeout)
                        for _ in range(nsamples)]
            except queue.Empty:
                if self.file_queue.started and not self.file_queue.filling and \
                   self.file_queue.qsize() == 0:
                    raise FileQueueDepleted('End of Training samples reached')
                else:
                    raise ValueError(
                        'Queue Empty Exception but File Queue is still' +
                        'active. Maybe the image loaders are under heavy load?')

        else:
            data = [catch_empty(lambda: self.get(block=block))
                    for _ in range(nsamples)]
            data = [x for x in data if type(x) is not queue.Empty]
        end = time.time()

        if len(data) == 0:
            if not self.file_queue.started:
                raise FileQueueNotStarted(
                    "Start the File Queue manager by calling " +
                    "FileQueue.load_epochs before calling get_batch")
            elif self.file_queue.started and not self.file_queue.filling and \
                    self.file_queue.qsize() == 0:
                raise FileQueueDepleted('End of Training samples reached')
            else:
                warnings.warn("No images in the image queue and get_batch " +
                              "was called with a non blocking request. " +
                              "Returning empty data")
                return None, None

        if self.epoch_size is not None:
            last_batch = (len(data) + self._read_count) >= self.epoch_size
            if last_batch:
                self._read_count = (len(data) + self._read_count -
                                    self.epoch_size)
                self._last_batch = True
            else:
                self._read_count += len(data)
                self._last_batch = False

        if self.logging_on:
            self._update_logger_info(end-start)

        # Unzip the data and labels before returning
        #  data, labels = zip(*data)
        labels = [x[1] for x in data]
        data = [x[0] for x in data]
        if labels[0] is None:
            return data, None
        else:
            return data, labels

    def _update_logger_info(self, fetch_time):
        idx = self.logger_info['call_count']
        idx += 1
        self.logger_info['call_count'] = idx

        # The summary scalars
        self.logger_info['av_qsize'] += self.qsize()
        self.logger_info['av_fetch_time'] += fetch_time
        if idx % self.write_period == 0:
            qsize = self.logger_info['av_qsize'] / self.write_period
            fetch_times = self.logger_info['av_fetch_time'] / self.write_period
            self.logger.log_scalar('queues/{}/fetch_time'.format(self.name),
                                   fetch_times, idx)
            self.logger.log_scalar('queues/{}/qsize'.format(self.name),
                                   qsize, idx)
            self.logger_info['av_qsize'] = 0
            self.logger_info['av_fetch_time'] = 0

        # The summary histograms
        # If we fill up the buffer, wrap around to start and write at beginning
        i = self.logger_info['epoch_idx']
        size = self.logger_info['epoch_size']
        i += 1
        size += 1
        if i % self.logger_info['epoch_qsize'].shape[0] == 0:
            i = 0
            size -= 1
        self.logger_info['epoch_qsize'][i] = self.qsize()
        self.logger_info['epoch_fetch_time'][i] = fetch_time

        if self._last_batch:
            self.logger.log_histogram(
                'queues/{}/fetch_time'.format(self.name),
                self.logger_info['epoch_fetch_time'][:size],
                idx, bins=1000)
            self.logger.log_histogram(
                'queues/{}/qsize'.format(self.name),
                self.logger_info['epoch_qsize'][:size],
                idx, bins=1000)
            i = 0
            size = 0

        self.logger_info['epoch_idx'] = i
        self.logger_info['epoch_size'] = size


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
        self._kill = False

    def get(self, block=True, timeout=0):
        if not self.started:
            raise FileQueueNotStarted(
                'Call load_epochs before trying to pull from the file queue')
        else:
            return super(FileQueue, self).get(block=block, timeout=timeout)

    def kill_loaders(self):
        """ Method to signal any threads that are filling this queue to stop.

        Threads will clean themselves up if the epoch limit is reached, but in
        case you want to kill them manually before that, you can signal them to
        stop here.
        """
        self._kill = True

    def load_epochs(self, files, shuffle=True, max_epochs=float('inf')):
        """
        Starts a thread to load the file names into the file queue.

        Parameters
        ----------
        files : list
            Can either be a list of filename strings or a list of tuples of
            (filenames, labels)
        shuffle : bool
            Whether to shuffle the list before adding it to the queue.
        max_epochs : int or infinity
            Maximum number of epochs to allow before queue manager stops
            refilling the queue.

        Notes
        -----
        Even if shuffle input is set to false, that doesn't necessarily mean
        that all images in the image queue will be in the same order across
        epochs. For example, if thread A pulls the first image from the
        list and then thread B gets the second 1. Thread A takes slightly
        longer to read in the image than thread B, so it gets inserted into
        the Image Queue afterwards.  Trying to synchronize across both queues
        could be done, but it would add unnecessary complications and overhead.

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
                kwargs={'files': myfiles, 'shuffle': shuffle}, daemon=True)
            self.thread.start()

    def manage_queue(self, files, shuffle=True):
        self.started = True
        self.filling = True
        self.epoch_count = 0
        self.epoch_size = len(files)

        while self.epoch_count < self.max_epochs and not self._kill:
            if self.qsize() < 0.5*len(files):
                epochs_to_put = min(
                    EPOCHS_TO_PUT, self.max_epochs - self.epoch_count)
                # Load multiple epochs in at a time
                for i in range(epochs_to_put):
                    if shuffle:
                        random.shuffle(files)
                    [self.put(item) for item in files]
                    self.epoch_count += 1
            else:
                time.sleep(5)

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

    def _load_image(self, im=''):
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
            raise FileQueueNotStarted(
                "File Queue has to be started before reading from it")

        while not self.iqueue._kill:
            # Try get an item - the file queue running out is the main way for
            # this thread to exit.
            try:
                item = self.fqueue.get_nowait()
                # Split the item into a filename and label
                try:
                    f, label = item
                except:
                    f = item
                    label = None

                img = self._load_image(os.path.join(self.base_dir, f))
                self.iqueue.put((img, label))
                self.fqueue.task_done()
            except queue.Empty:
                if not self.fqueue.filling:
                    return

        if self.iqueue._kill:
            self.fqueue.kill_loaders()


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
