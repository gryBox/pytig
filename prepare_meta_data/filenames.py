import os
import sys

import pandas as pd
import textacy


import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# class PrepareFilenames():
#     """
#     Prepares base file names to match between text and image directories
#     """


def extract_basename(df, flpthColNms_lst=['images','text']):
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





def get_data_basenames(flpth):
    """
    Lowercases the basename for a fileoath
    """
    basename = os.path.splitext(os.path.basename(flpth))[0]

    return basename







