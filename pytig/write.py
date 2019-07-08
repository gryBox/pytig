"""
Loads datasets to train algorithms and generate images for supported text to image algorithms.


"""

import sys
import os
import shutil
import requests
from io import BytesIO
from zipfile import ZipFile
import pickle

import pandas as pd


import textacy
import en_core_web_sm

# Load english language model for sents parsing and caption relabeling
en = en_core_web_sm.load()

from pytig import read
import pytig as ptg

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
    else:
        # Delete metadata folder s-- should always be the top level directory
        shutil.rmtree(os.path.join(data_dir_path, zipfile.namelist()[0]))

    # TODO: (?) Add .gitignore to the data dir immediatly


    # Write zip data to dir for preparing metadat files
    zipfile.extractall(data_dir_path)

    logging.debug(zipfile.printdir())

    return

def filenames_to_df(image_dir_path, text_dir_path, txt_ext=".txt", img_ext=".jpg"):
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

        # Handle what extentenstions
        if dirName is txtDirName:
            extension = txt_ext
        else:
            extension = img_ext

        # Only execute if path is a real path
        if os.path.exists(filepath):

            flNm_lst = list(textacy.io.utils.get_filepaths(
                filepath,
                extension=extension,
                ignore_invisible=True,
                recursive=True))

            flNm_lst.sort()
            file_names_dict[dirName] = flNm_lst

    # Make a df with all the filenames
    filenames_df = pd.DataFrame(file_names_dict)#.sort_values(by=list(flnameDir_dict.keys()))


    return filenames_df

def txt_to_docrec(flpth):
    # load txtfile
    txt_str = next(textacy.io.text.read_text(flpth, mode='rt', encoding=None, lines=False))

    # Make a record (text, metatdata)
    metadata = {
        "filepath": flpth,
        "filename": os.path.splitext(os.path.basename(flpth))[0]
        }

    record = (txt_str, metadata)

    return record

def txt_to_corpus(txt_dir, lang=en , txt_extention=".txt"):
    """
    Reads a text directory and puts in a textacy corpus with the filename as metadata
    and adds a docstats_df
    """
    # Get the name of the function - should be decorator for every function
    functionNameAsString = sys._getframe().f_code.co_name
    logging.debug(f"Function: {functionNameAsString} -- Loading Text from: {txt_dir}")

    # Get a list of files to get text for
    flpth_gen = textacy.io.utils.get_filepaths(txt_dir,
                               match_regex=None,
                               ignore_regex=None,
                               extension=".txt",
                               ignore_invisible=True,
                               recursive=True)

    # Loop throuh the text directory (input), for all the files ending with .txt
    #docs_lst = [dask.delayed(txtfile_to_doc)(flpth, en) for flpth in flpth_gen]
    rec_lst = [txt_to_docrec(flpth) for flpth in flpth_gen]
    #docs = dask.compute(docs_lst)

    # Add docs to a spacy corpus
    crps = textacy.Corpus(en, rec_lst)

    # Calculate stats for each doc in a corpus and make a docstats_df
    crpsStats = ptg.corpus_stats.CorpusStats(crps)

    crps.docstats_df = crpsStats.docstats_df

    logging.debug(f"Function: {functionNameAsString} -- Loaded {crps.n_docs}")

    return crps

def df_to_corpus(df, txt_column='RESOURCE'):
    # Load into textacy to delimit sentences
    img_labels = df.to_dict(orient="records")
    text_stream, metadata_stream = textacy.io.split_records(img_labels, txt_column)

    # Load english model

    corpus = textacy.Corpus(lang=en, texts=text_stream, metadatas=metadata_stream)

    return corpus

def obj_to_pickle(obj, flpth):

    logging.info(f"Writing pickle file to {flpth}")
    logging.debug(f"Writing pickle file of file type: {type(obj)}")

    if not os.path.exists(flpth):
        os.mkdir(os.path.dirname(flpth))

    with open(flpth, 'wb') as f:
        pickle.dump(obj, f)

    return


