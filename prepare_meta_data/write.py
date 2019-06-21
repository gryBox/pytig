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

import prepare_meta_data as pmd


def zip_to_metadata_dir(zip_url, data_dir_path):
    """
    Extracts a zipfile to a data directory path.
    Returns a zipfile object
    """
    # Returns a zipped directory
    zipfile = pmd.read.zip_from_url(zip_url)

    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    # Write zip data to dir for preparing metadat files
    zipfile.extractall(data_dir_path)

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

def df_to_corpus(df, text_column='text'):
    pass