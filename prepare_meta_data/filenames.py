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

def check_equal_len(filename1_lst, filename2_lst):
    """
    Input two file name lists and check for diff

    """
    if len(filename1_lst)!=len(filename2_lst):
        logging.info(f"Error: Number of text files and image files do not match")

        return False
    else:

        return True





