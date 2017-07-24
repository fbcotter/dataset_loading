.. _ImageQueue-label:

ImageQueue 
==========

The Image Queue is the interface between the package and your main program. 
Once you have built a file queue to store the file names to read in, you can
create an ImageQueue. Standard would look like this::

    import dataset_loading as dl
    file_queue = dl.FileQueue()
    file_queue.load_epochs(<list_of_files>, max_epochs=50)
    img_queue = dl.ImgQueue()
    img_queue.start_loaders(file_queue, num=3)
    # Wait for the image queue to fill up
    sleep(5)
    img_queue.get_batch(<batch_size>)

Calling the start_loaders method spins up <num> threads to pull from the file
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

E.g.::
    
    def norm_image(x):
        adjusted_stddev = max(np.std(x), 1.0/np.sqrt(x.size))
        return (x-np.mean(x))/adjusted_stddev
    imsize = (224,224)
    path_offset = '/scratch/share/pascal'
    img_queue.start_loaders(file_queue, num=3, img_size=imsize, 
                            img_dir=path_offset, transform=norm_image)


For more info on the ImgQueue, see its 
:py:class:`docstring <dataset_loading.core.ImgQueue>`.
