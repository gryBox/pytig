"""
Loads datasets to train algorithms and generate images for supported text to image algorithms.


"""
import pandas as pd
import sys
import os
import shutil
import requests
from io import BytesIO
from zipfile import ZipFile
import glob

import textacy
import en_core_web_sm

from prepare_meta_data import read

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def zip_to_metadata_dir(zip_url, data_dir_path):
    """
    Extracts a zipfile to a data directory path.
    Returns a zipfile object
    """
    # Returns a zipped directory
    zipfile = read.zip_from_url(zip_url)

    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    # Write zip data to dir for preparing metadat files
    zipfile.extractall(data_dir_path)

    return zipfile


def txt_to_corpus(txt_dir, txt_extention=".txt"):
    """
    Reads a text directory and puts in a textacy corpus with the filename as metadata
    """
    # Get the name of the function - should be decorator for every function
    functionNameAsString = sys._getframe().f_code.co_name


    # Load english language model for sents parsing and caption relabeling
    en = en_core_web_sm.load()

    # Initalize spacy corpus using textacy
    imageLabels_corpus = textacy.corpus.Corpus(en)

    # Loop throuh the text directory (input), for all the files ending with .txt
    search_dir = f"{txt_dir}/*{txt_extention}"
    logging.debug(f"Function: {functionNameAsString} -- Loading Text from: {search_dir}")
    for f in glob.glob(search_dir):
        txt_str = open(f).read()
        imageLabels_corpus.add_record((txt_str, {"file_name": f})  )

    logging.debug(f"Function: {functionNameAsString} -- Loaded {imageLabels_corpus.n_docs}")

    return imageLabels_corpus

def df_to_corpus(df, text_column='text'):
    pass