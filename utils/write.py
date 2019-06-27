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


# Load english language model for sents parsing and caption relabeling
en = en_core_web_sm.load()

from . import read

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

    # Check if data dir exists
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    # TODO: (?) Add .gitignore to the data dir immediatly


    # Write zip data to dir for preparing metadat files
    zipfile.extractall(data_dir_path)

    return zipfile

def filenames_to_df(image_dir_path, text_dir_path):
    """
    Loads file names from text and image directory
    """
    # Informational: Get the name of the function - should be decorator for every function
    functionNameAsString = sys._getframe().f_code.co_name

    # 1. Define filename col names
    txtDirName = os.path.basename(text_dir_path)
    imgDirName = os.path.basename(image_dir_path)

    flnameDir_dict={
        txtDirName: text_dir_path,
        imgDirName: image_dir_path
    }
    # 2. Load filenames into a pandas data frame for transformation and check
    logging.info(f"{functionNameAsString} -- Loading Filenames from {flnameDir_dict.keys()} ")

    file_names_dict = dict()
    for dirName, filepath in flnameDir_dict.items():

        # Only execute if path is a real path
        if os.path.exists(filepath):

            file_names_dict[dirName] = list(textacy.io.utils.get_filepaths(
                filepath,
                extension=None,
                ignore_invisible=True,
                recursive=True))

    # Make a df with all the filenames
    filenames_df = pd.DataFrame(file_names_dict)
    filenames_df['ImagePath'] = image_dir_path
    filenames_df['TxtPath'] = text_dir_path

    return filenames_df

def txt_to_corpus(txt_dir, crps_file_tag='file_name' , txt_extention=".txt"):
    """
    Reads a text directory and puts in a textacy corpus with the filename as metadata
    """
    # Get the name of the function - should be decorator for every function
    functionNameAsString = sys._getframe().f_code.co_name




    # Initalize spacy corpus using textacy
    imageLabels_corpus = textacy.corpus.Corpus(en)

    # Loop throuh the text directory (input), for all the files ending with .txt
    search_dir = f"{txt_dir}/*{txt_extention}"
    logging.debug(f"Function: {functionNameAsString} -- Loading Text from: {search_dir}")
    for f in glob.glob(search_dir):
        txt_str = open(f).read()
        imageLabels_corpus.add_record((txt_str, {f"{crps_file_tag}": f})  )

    logging.debug(f"Function: {functionNameAsString} -- Loaded {imageLabels_corpus.n_docs}")

    return imageLabels_corpus

def df_to_corpus(df, txt_column='RESOURCE'):
    # Load into textacy to delimit sentences
    img_labels = df.to_dict(orient="records")
    text_stream, metadata_stream = textacy.io.split_records(img_labels, txt_column)

    # Load english model

    corpus = textacy.Corpus(lang=en, texts=text_stream, metadatas=metadata_stream)

    return corpus