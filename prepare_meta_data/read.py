"""
Loads datasets to train algorithms and generate images for supported text to image algorithms.


"""
from collections import namedtuple
import pandas as pd
import os
import shutil
import requests
from io import BytesIO
from zipfile import ZipFile
import textacy

import glob
import en_core_web_sm


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
        data_home = os.path.join('~', 'pytig/data') #TODO: Generalize to pytig
    data_home = os.path.expanduser(data_home)

    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home

def zip_from_url(file_url):
    """
    Get a zipped file from a url
    """
    url = requests.get(file_url)
    zipfile = ZipFile(BytesIO(url.content))

    return zipfile

def txt_to_corpus(txt_dir):
    """
    Reads a text directory and puts in a textacy corpus with the filename as metadata
    """
    en = en_core_web_sm.load()
    imageLabels_corpus = textacy.corpus.Corpus(en)

    for f in glob.glob(txt_dir + "*.txt"):
        txt_str = open(f).read()
        imageLabels_corpus.add_record((txt_str, {"file_name": f})  )

    return imageLabels_corpus


def photosynthesis_raw(data_home=None, data_url="https://github.com/gryBox/pytig-data/raw/master/photosynthesis_raw.zip"):
    """
    Load raw photosynthesis data that is split into image and text directories. i.e text needs to be processed

    """

    # data_lst = lds.get_url_zip(data_url)

    # # Determine where to put the loaded data.
    # dataDir_flpth = get_data_home(data_home)
    pass



def photosynthesis_lbld():
    pass