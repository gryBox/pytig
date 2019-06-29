import os
import sys

import pandas as pd
import textacy

from pytig import write

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class PrepareFilenames():
    """
    Extracts filenames for metadata folder
    - renames filenams
    - writes filename.txt to metadata folder
    """
    def __init__(self, image_training_data_flpth, text_training_data_flpth):

        # File paths to data directories
        self.image_training_data_flpth = image_training_data_flpth
        self.text_training_data_flpth = text_training_data_flpth

        self.txt_dir = os.path.basename(self.text_training_data_flpth)
        self.img_dir = os.path.basename(self.image_training_data_flpth)

        # Load file names to df from input directories
        self.fileNames_df = write.filenames_to_df(self.image_training_data_flpth, self.text_training_data_flpth)

    def extract_basename(self, df, flpthColNms_lst=['images','text']):
        """Extracts the base file names from the txt and image filepaths to a list"""

        # Stack the text and column images flpth to one column
        flpths_df = df[flpthColNms_lst].stack()
        #print(type(flpths_df))

        # extract the basenames for each of the directories
        baseNm_df = flpths_df.apply(lambda x: os.path.splitext(os.path.basename(x))[0]).unstack()

        # # Merge to one column if the basenames are the same
        error = [1,0]
        baseNm_series = baseNm_df.apply(lambda x: x.unique()[0] if x.nunique()==1 else error, axis=1)

        return baseNm_series


    def rename(self, flpth):
        """
        Lowercases the basename for a fileoath
        """
        basename = os.path.splitext(os.path.basename(flpth))[0]

        return basename









