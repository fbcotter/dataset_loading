.. _ImageQueue-label:

ImageQueue 
==========

The Image Queue is the interface between the package and your main program. 
Once you have built a file queue to store the file names to read in, you can
create an ImageQueue. Standard would look like this:

.. code:: python

    import dataset_loading as dl
    file_queue = dl.FileQueue()
    file_queue.load_epochs(<list_of_files>, max_epochs=50)
    img_queue = dl.ImgQueue()
    img_queue.start_loaders(file_queue, num_threads=3)
    # Wait for the image queue to fill up
    sleep(5)
    img_queue.get_batch(<batch_size>)

Calling the start_loaders method spins up <num_threads> threads to pull from the file
queue and write to the image queue. See the 
:py:meth:`ImgQueue.start_loaders <dataset_loading.core.ImgQueue.start_loaders>` 
docstring for more info on the parameters you have here, but note that this is
where you set:

- Path offsets for the files in the file queue (in case the files in the 
  file queue weren't the absolute path of the images). 
- The size of the image to resize to. By default (a parameter of None), no 
  resizing will be done. 
- Any pre-filtering operation to be done to the images (e.g. contrast 
  normalization). 

E.g.:
    
.. code:: python

    def norm_image(x):
        adjusted_stddev = max(np.std(x), 1.0/np.sqrt(x.size))
        return (x-np.mean(x))/adjusted_stddev
    imsize = (224,224)
    path_offset = '/scratch/share/pascal'
    img_queue.start_loaders(file_queue, num_threads=3, img_size=imsize, 
                            img_dir=path_offset, transform=norm_image)


For more info on the ImgQueue, see its 
:py:class:`docstring <dataset_loading.core.ImgQueue>`.

Note
----
By default the `get_batch` function does NOT block. I.e. if you call it, asking
for 100 samples but only 50 are available, it will return with 50. If you do
not want this, then you can set the parameter `block=True`. You may also
consider setting the `timeout` parameter to a sensible value.

.. _ImageQueue-monitoring-label:

Queue Monitoring
----------------
You can take advantage of tensorflow's tensorboard and plot out some queue
statistics too. The dataset_loading package is meant to be able to work without
tensorflow, so attempting these methods may throw warnings and not work. Logging
is automatically done when calls to the `get_batch` method are made.

.. code:: python
    
    img_queue.start_loaders(file_queue, num_threads=3, transform=preprocess)
    file_writer = tf.summary.FileWriter('./log', tf.get_default_graph())
    # Write period is the sample period in numbers of batches for dumping data
    img_queue.add_logging(file_writer, write_period=10)

.. _ImageQueue-properties-label:

Properties
----------
Here are some useful properties of the ImgQueue class that may help you in
designing your program:

- last_batch : True if the previously read batch was the last in the epoch.
  Reading this value resets it to false.
- epoch_size : The number of images in the epoch. Interpreted from the File
  Queue. Cannot always determine this.
- read_count : How many images have been read in the current epoch
- image_shape : Inspects the queue and gets the shape of the images in it.
  Useful to check what the output shape from any preprocessing steps done
  beforehand were.
- label_shape : Inspects the queue and gets the shape of the labels in it.
