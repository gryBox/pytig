"""
Loads datasets to train algorithms and generate images for supported text to image algorithms.


"""
from collections import namedtuple
import pandas as pd
import os
import shutil

#import


RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])


def get_data_home(data_home=None):
    """Return the path of the pytig data dir.
    This folder is used by some large dataset loaders to avoid downloading the
    data several times.
    By default the data dir is set to a folder named 'data' in the
    user home folder.

    ----------
    data_home : str | None
        The path to pytig data dir.
    """
    if data_home is None:
        data_home = os.environ.get('DATA',
                                os.path.join('~', 'pytig/data')) #TODO: Generalize to pytig
    data_home = os.path.expanduser(data_home)

    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home

def photosynthesis_raw(data_home=None):

    # Determine where to put the loaded data.
    dataDir_flpth = get_data_home(data_home)

    # Move example data to data directory
    shutil.copy("/home/ismail/Downloads/", data_home)

    return

def photosynthesis_lbld():
    pass