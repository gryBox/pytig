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




def zip_from_url(file_url):
    """
    Get a zipped file from a url
    """
    url = requests.get(file_url)
    zipfile = ZipFile(BytesIO(url.content))

    return zipfile




def photosynthesis_lbld():
    pass