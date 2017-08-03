from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
from dataset_loading import core


def img_sets():
    """
    List all the image sets from Pascal VOC. Don't bother computing
    this on the fly, just remember it. It's faster.
    """
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


def img_dict(base_dir):
    d = {}
    for i, cat in enumerate(img_sets()):
        filename = os.path.join(base_dir, 'ImageSets', 'Main',
                                cat+'_trainval.txt')
        df = pd.read_csv(filename, delim_whitespace=True, header=None,
                         names=['filename', 'true'])
        df = df[df['true'] == 1]
        files = df['filename'].values
        for f in files:
            if f in d.keys():
                d[f].append(i)
            else:
                d[f] = [i]
    return d


def load_pascal_data(data_dir, max_epochs=None, thread_count=3,
                     imsize=(128,128)):
    """Will use a filename queue and img_queue and load the data
    """
    file_queue = core.FileQueue()
    #  d = img_dict(data_dir)

    img_queue = core.ImageQueue(files_in_epoch=250, maxsize=1000)
    threads = []
    for i in range(thread_count):
        thread = core.imLoader('Loader ' + str(i+1), file_queue, img_queue,
                               imsize, data_dir)
        thread.start()
        threads.append(thread)

    return img_queue
