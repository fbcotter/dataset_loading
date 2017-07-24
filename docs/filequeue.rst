.. _FileQueue-label:

FileQueue 
=========

Typically, you will set up a file queue to give to an Image Loader thread and
will never need to touch it, but if you do wish to use it directly, there are
some things to note. For these notes, it is useful to look at the typical
usage::

    import dataset_loading as dl
    files = [<some list of filenames or a list of tuples of (filenames, labels)]
    file_queue = dl.FileQueue()
    file_queue.load_epochs(files, max_epochs=50)

Calling the load_epochs function actually spins up a thread to manage the file
queue. This thread doesn't have to do much (so we only use 1), but it will
refill the queue if it starts to get too low (<50% of one epoch). Initially, it
will load 10 epochs worth of the <files> list into the FileQueue. This is not
too important a quantity, we just want it to be big enough so that calls to the
FileQueue.get() shouldn't be blocking most of the time, and not so big that the
FileQueue takes up lots of memory. 

In case you happen to request a lot of files when the queue is relatively
empty, it would be a good idea to put a small timeout on the get(). Not so long
(as you may have hit the end of the epoch limit and the queue will not refill!)
and long enough to allow the FileQueue manager thread to detect the queue has
emptied and give it time to fill up. Perhaps 10ms should work, i.e.::

    list_of_files = [file_queue.get(timeout=0.01) for _ in range(1000)]
