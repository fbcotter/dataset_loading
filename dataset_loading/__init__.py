from .core import ImgQueue, FileQueue
from .core import ImgQueueNotStarted, FileQueueNotStarted, FileQueueDepleted

__author__ = "Fergal Cotter"
__version__ = "0.0.4"
__version_info__ = tuple([int(d) for d in __version__.split(".")])

__all__ = ['ImgQueueNotStarted', 'FileQueueNotStarted',
           'FileQueueDepleted', 'ImgQueue', 'FileQueue']
