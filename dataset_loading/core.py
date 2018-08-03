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

__all__ = ['ImgQueue', 'FileQueue', 'ImgLoader', 'FileQueueNotStarted',
           'FileQueueDepleted', 'ImgQueueNotStarted']

EPOCHS_TO_PUT = 10
FILEQUEUE_SLEEPTIME = 5
FILEQUEUE_BLOCKTIME = 1
IMGQUEUE_BLOCKTIME = 3


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
        self._epoch_count = 0
        self._last_batch = False
        self.in_memory = False
        self.name = name
        self.logging_on = False
        self._kill = False
        self.file_queue = None

        self.loaders_alive = []
        self.loaders_started = False

    def __repr__(self):
        def bool2str(x):
            if x:
                return "yes"
            else:
                return "no"
        return ("ImgQueue instance - {}.\n".format(self.name) +
                "Loaders started: {}\n".format(bool2str(self.loaders_started)) +
                "Loaders done: {}\n".format(bool2str(self.loaders_finished)) +
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
        if self.file_queue is not None:
            return self.file_queue.epoch_size
        else:
            return -1

    @property
    def read_count(self):
        """ Returns how many images have been read from this queue.
        """
        return self._read_count

    @property
    def epoch_count(self):
        """ Returns what epoch we are currently at """
        return self._epoch_count

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
    def killed(self):
        """ Returns True if the queue has been asked to die. """
        return self._kill

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

    @property
    def filling(self):
        """ Returns true if the file queue is being filled """
        return (True in self.loaders_alive)

    @property
    def loaders_finished(self):
        return self.loaders_started and not self.filling

    def join(self):
        """ Method to signal any threads that are filling this queue to stop.

        Threads will clean themselves up if the epoch limit is reached, but in
        case you want to kill them manually before that, you can signal them to
        stop here. Note that if these threads are blocked waiting on input, they
        will still stay alive (and blocked) until whatever is blocking them
        frees up. This shouldn't be a problem though, as they will not be taking
        up any processing power.

        If there is a file queue associated with this image queue, those threads
        will be stopped too.

        Note: Overloads the queue join method which normally blocks until the
        queue has been emptied. This will return even if the queue has data in
        it.
        """
        # Kill the file queue
        if self.file_queue is not None:
            self.file_queue.join()

        # Tell the loaders to stop filling the queue
        self._kill = True

        # Empty the queue once through - note that the loaders may still fill it
        # up afterwards, but we want to just stop them blocking trying to put
        # into a full queue
        while(self.filling):
            try:
                self.get_nowait()
            except queue.Empty:
                time.sleep(0.01)

        for l in self.loaders:
            l.join()

    def start_loaders(self, file_queue, num_threads=3, img_dir=None,
                      img_size=None, transform=None):
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
        self.loaders_alive = [True,] * num_threads
        loaders = [
            ImgLoader(i, file_queue, self, img_dir=img_dir,
                      img_size=img_size, transform=transform)
            for i in range(num_threads)
        ]
        [loader.start() for loader in loaders]
        self.loaders_started = True
        self.loaders = loaders

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
        # Create a file queue. This will only contain indices into the numpy
        # arrays data and labels.
        self.file_queue = FileQueue()
        files = list(range(data.shape[0]))
        self.file_queue.load_epochs(files, shuffle=shuffle,
                                    max_epochs=max_epochs)

        loaders = []
        #  self.loaders_alive = self.mngr.list([True,]*num_threads)
        self.loaders_alive = [True,] * num_threads
        for i in range(num_threads):
            thread = threading.Thread(
                target=_mini_loader, name='Mini Loader Thread',
                kwargs={'idx': i,
                        'fq': self.file_queue,
                        'iq': self,
                        'data': data,
                        'labels': labels,
                        'transform': transform},
                daemon=True)
            thread.start()
            loaders.append(thread)
        self.loaders_started = True
        self.loaders = loaders

    def get(self, block=True, timeout=None):
        """ Get a single item from the Image Queue"""
        if not self.loaders_started:
            raise ImgQueueNotStarted(
                "Start the Image Queue Loaders by calling start_loaders " +
                "before calling get")
        else:
            data = super(ImgQueue, self).get(block, timeout)
            self._read_count += 1
            return data

    def get_batch(self, batch_size, timeout=IMGQUEUE_BLOCKTIME):
        """Tries to get a batch from the Queue.

        If there is less than a batch of images, it will grab them all.
        If the epoch size was set and the tracking counter sees there are
        fewer than <batch_size> images until we hit an epoch, then it will
        cap the amount of images grabbed to reach the epoch.

        Parameters
        ----------
        batch_size : int
            How many samples we want to get.
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
        if not self.loaders_started:
            raise ImgQueueNotStarted(
                "Start the Image Queue Loaders by calling start_loaders " +
                "before calling get")
        else:
            return self._get_batch(batch_size, timeout)

    def _get_batch(self, batch_size, timeout=None):
        # Determine some limits on how many images to grab.
        rem = batch_size
        if self.epoch_size is not None:
            rem = self.epoch_size - (self._read_count % self.epoch_size)

        nsamples = min(rem, batch_size)
        start = time.time()
        data = []
        for i in range(nsamples):
            try:
                item = self.get(block=True, timeout=timeout)
            except queue.Empty:
                time.sleep(FILEQUEUE_BLOCKTIME)
                if self.loaders_finished:
                    raise FileQueueDepleted("No more images left")
                else:
                    # Allow some time for the file queues to definitely finish
                    # before raising an exception
                    raise ValueError(
                        'Queue Empty Exception but File Queue is still' +
                        'active. Maybe the image loaders are under heavy ' +
                        'load?')
            data.append(item)
        end = time.time()

        if self.epoch_size is not None:
            if self._read_count // self.epoch_size > self._epoch_count:
                self._last_batch = True
                self._epoch_count = self._read_count // self.epoch_size

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


def _mini_loader(idx, fq, iq, data, labels, transform):
    """ Image Queue manager for when we have a dataset provided

    If the dataset is small enough to fit into memory and we don't need file
    queues, we can use an image queue and a simpler multithreaded function to
    load it. In this case, we make a mock file queue, which is just filled with
    indices into the np array of images.
    """
    #  print("Starting miniloader thread {}".format(idx))
    if labels is not None:
        assert data.shape[0] == labels.shape[0]

    while not iq.killed:
        # Try get an item
        #  print('reading fq')
        have_data = False
        try:
            #  item = self.file_queue.get_nowait()
            item = fq.get(block=True, timeout=FILEQUEUE_BLOCKTIME)
            have_data = True
        except queue.Empty:
            # If the file queue ran out, exit quietly
            if not fq.filling:
                break
        except OSError:
            # Perhaps the file queue has been shut?
            break

        if have_data:
            #  print('inserting into iq')
            try:
                assert not isinstance(item, tuple)
            except AssertionError:
                item = item[0]

            # 'Load' the image and label - reshape if necessary
            img = data[item]
            if transform is not None:
                img = transform(img)
            if labels is not None:
                label = labels[item]
            else:
                label = None

            # Put it into my queue.
            iq.put((img, label))
    #  print("Ending processing thread {}".format(idx))
    #  iq.cancel_join_thread()
    iq.loaders_alive[idx] = 0


class FileQueue(queue.Queue):
    """A queue to hold filename strings

    This queue is used to indicate what order of jpeg files should be read. It
    may also be a good idea to put the class label alongside the filename as a
    tuple, so the main program can get access to both of these at the same time.

    Create the class, and then call the load_epochs() method to start a thread
    to manage the queue and refill it as it gets low.

    The maxsize is not provided as an option as we want the queue to be able to
    take entire epochs and not be restricted on the upper limit by a maxsize.
    The data should be no problem as the queue entries are only integers.
    """
    def __init__(self, maxsize=0):
        queue.Queue.__init__(self, maxsize=maxsize)
        self._epoch_size = -1
        self.thread = None
        self.started = False

        # Flags for the ImgQueue
        self.started = False
        self._kill = False
        self.loader_alive = False
        self._epoch_count = -1

    @property
    def epoch_count(self):
        """ The current epoch count """
        return self._epoch_count

    @epoch_count.setter
    def epoch_count(self, val):
        """ Update the epoch count """
        self._epoch_count = val

    @property
    def killed(self):
        """ Returns true if the queue has been asked to die """
        return self._kill

    @property
    def filling(self):
        """ Returns true if the file queue is being filled """
        return self.loader_alive

    @property
    def epoch_size(self):
        """ Gives the size of one epoch of data """
        return self._epoch_size

    def get(self, block=True, timeout=None):
        """ Get a single item from the Image Queue"""
        if not self.started:
            raise FileQueueNotStarted(
                "Start the File Queue manager by calling " +
                "FileQueue.load_epochs before calling get_batch")
        else:
            return super(FileQueue, self).get(block, timeout)

    def _depleted(self):
        raise FileQueueDepleted('End of Training samples')

    def join(self):
        """ Method to signal any threads that are filling this queue to stop.

        Threads will clean themselves up if the epoch limit is reached, but in
        case you want to kill them manually before that, you can signal them to
        stop here.

        Note: Overloads the queue join method which normally blocks until the
        queue has been emptied. This will return even if the queue has data in
        it.
        """
        # Tell the loaders to stop filling the queue
        self._kill = True
        time.sleep(1)

        # Ensure the file loaders aren't blocking by eating up the remaining
        # data
        while(self.loader_alive):
            try:
                self.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
        self.thread.join()

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
            self.thread = threading.Thread(
                target=file_loader, name='File Queue Thread',
                kwargs={'files': myfiles, 'fq': self, 'shuffle': shuffle},
                daemon=True)
            self.max_epochs = max_epochs
            self._epoch_count = 0
            self._epoch_size = len(files)
            self.loader_alive = True
            self.started = True
            self.thread.start()


def file_loader(files, fq, shuffle=True):
    """ Function to fill up the file queue. This tops up the file queue whenever
    it gets low.
    """
    while fq.epoch_count < fq.max_epochs and not fq.killed:
        #  print('Checking fq')
        if fq.qsize() < 0.5*len(files) and fq.epoch_count < fq.max_epochs:
            epochs_to_put = min(
                EPOCHS_TO_PUT, fq.max_epochs - fq.epoch_count)
            # Load multiple epochs in at a time
            for i in range(epochs_to_put):
                if shuffle:
                    random.shuffle(files)
                [fq.put(item) for item in files]
                fq.epoch_count += 1
        else:
            #  print('Sleeping')
            time.sleep(FILEQUEUE_SLEEPTIME)
    fq.loader_alive = False
    #  fq.cancel_join_thread()


class ImgLoader(threading.Thread):
    """ A thread to load in images from a filename queue into an image queue.

    This must be instantiated with a file queue (to read from) and an image
    queue (to load into).

    If you call the :py:meth:`ImageQueue.kill` method, the signal will be passed
    on to this process and it will die gracefully.
    """
    def __init__(self, idx, file_queue, img_queue, img_size=None,
                 img_dir=None, transform=None):
        threading.Thread.__init__(self, daemon=True)
        self.idx = idx
        self.fqueue = file_queue
        self.iqueue = img_queue
        self.img_size = img_size
        if img_dir is not None:
            self.base_dir = img_dir
        else:
            self.base_dir = ''
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
        #  print("Starting ImgLoader {}".format(self.idx))
        # Check the image queue's kill signal. So long as this is down, keep
        # running.
        while not self.iqueue.killed:
            # Try get an item - the file queue running out is the main way for
            # this thread to exit.
            data = False
            try:
                item = self.fqueue.get(block=True, timeout=FILEQUEUE_BLOCKTIME)
                data = True
            except queue.Empty:
                # If the file queue timed out and there's nothing filling it -
                # exit
                if not self.fqueue.filling:
                    break

            # Split the item into a filename and label
            if data:
                try:
                    f, label = item
                except:
                    f = item
                    label = None

                img = self._load_image(os.path.join(self.base_dir, f))
                self.iqueue.put((img, label))
        #  print("Img Loader {} Done".format(self.idx))
        #  self.iqueue.cancel_join_thread()
        self.iqueue.loaders_alive[self.idx] = 0


def catch_empty(func, handle=lambda e: e, *args, **kwargs):
    """ Returns the empty exception rather than raising it

    Useful for calling queue.get in a list comprehension
    """
    try:
        return func(*args, **kwargs)
    except queue.Empty as e:
        return handle(e)


class ImgQueueNotStarted(Exception):
    """Exception Raised when trying to pull from an Image queue that hasn't had
    its feeders started.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class FileQueueNotStarted(Exception):
    """Exception Raised when trying to pull from a File queue that hasn't had
    its manager started."""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class FileQueueDepleted(Exception):
    """Exception Raised when the file queue has been depleted. Will be raised
    when the epoch limit is reached."""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
