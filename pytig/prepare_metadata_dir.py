import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from pytig import filenames
from pytig import write

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class  Metadata():
    """
    Purpose: Prepares a metadata directory for the text to image AttnGAN algorithim

    Requirements: A directory with corresponding labeled images and text files.
    Example data directory.
    - data
        - photosynthesis
            --text
                file_name.txt
                file_name1.txt
            --images
                file_name.png
                file_name1.png

    Inputs:  metadata_flpth - flpth to the training data directory i.e metadata_dir
    Output:  A metadata folder with:
                -- "filenames.txt" containing the base filenames for text and image data
                -- "train" dir with list of filenames to use for training
                    -- "filenames.pickle" - Contains all the file names to train on
                -- "test" dir with list of  filenames for cross validation/prediction
                    -- "filenames.pickle" - Contains all the file names to test on
    Notes:
        -  Captions do not have to be lowercase or unpunctuated
        -  It is helpful to normalize hyphenated words carbon-dioxide -> carbon dioxide.
    """
    def __init__(self, metadata_flpth, image_data_flpth, text_data_flpth, **kwargs):

        # User Inputs
        self.metadata_flpth = metadata_flpth
        self.image_data_flpth = image_data_flpth
        self.text_data_flpth = text_data_flpth

        # Training and Test filepaths
        self.train_dir_name = kwargs.setdefault('train_dir_name', "train")
        self.train_dir_flpth = os.path.join(self.metadata_flpth, self.train_dir_name)

        self.test_dir_name = kwargs.setdefault('train_dir_name', "test")
        self.test_dir_flpth = os.path.join(self.metadata_flpth, self.test_dir_name)

        # Prepare filenames and write file basename to filenames.tx
        self.txt_ext = kwargs.setdefault('txt_ext', ".txt")
        self.img_ext = kwargs.setdefault('img_ext', ".jpg")
        self.preprocess_text = kwargs.setdefault("preprocess_text", True)
        self._enumerate = kwargs.setdefault("preprocess_text", False)

        self.lowercase=kwargs.setdefault('lowercase', True),
        self.no_urls=kwargs.setdefault('no_urls', True),
        self.no_emails=kwargs.setdefault('no_emails', True),
        self.no_phone_numbers=kwargs.setdefault('no_phone_numbers', True),
        self.no_numbers=kwargs.setdefault('no_numbers', False),
        self.no_currency_symbols=kwargs.setdefault('no_currency_symbols', True),
        self.no_punct=kwargs.setdefault('no_punct', False),
        self.no_contractions=kwargs.setdefault('no_contractions', True),
        self.no_accents=kwargs.setdefault('no_accents', True)

        # self.preparedFilenames = filenames.PrepareFilenames(self.metadata_flpth,
        # self.image_data_flpth, self.text_data_flpth)


        # self.preparedFilenames.rename_basename(preprocess_text=self.preprocess_text, _enumerate=self._enumerate, **kwargs)


        #preparedFilenames.basenames_to_txt



    def split_data(self, filename_df, split_ratio=0.3, filenames_clm='filename'):
        """
        Splits filenames between training and cross validation
        and write to directories test and train

        """

        train_filenames, test_filenames = train_test_split(filename_df[filenames_clm], test_size=0.3)

        logging.info("Number of training data files: {train_filenames.shape}")
        logging.info("Number of test data files: {test_filenames.shape}")

        # Write filenames as a to train directory and test directory as pickled lists
        filenames_pickle_nm = "filenames.pickle"
        write.obj_to_pickle(train_filenames.to_list(), os.path.join(self.metadata_flpth, 'train', filenames_pickle_nm))
        write.obj_to_pickle(test_filenames.to_list(),  os.path.join(self.metadata_flpth, 'test', filenames_pickle_nm))

        return train_filenames, test_filenames