from .core import *  # noqa
import os

# Imports the __version__ variable
exec(open(os.path.join(os.path.dirname(__file__), '../version.py')).read())
__author__ = "Fergal Cotter"
__version_info__ = tuple([int(d) for d in __version__.split(".")])  # noqa

__all__ = ['core']
