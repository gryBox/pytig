import os
import sys

import pandas as pd
import argparse

from pytig import write

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser(description='Normalize, split (training, eval), and other filename preperations for the AttnGAN algorithim')
    # parser.add_argument('--cfg', dest='cfg_file',
    #                     help='optional config file',
    #                     default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--metadata_flpth', type=str)
    parser.add_argument('--image_data_flpth', default='')
    parser.add_argument('--text_data_flpth', default='')
    parser.add_argument('--basename_clmn', type=str, default='basename')
    parser.add_argument('--filepaths_clmn', type=str, default='filepaths')
    parser.add_argument('--txt_ext', type=str, default='txt' )
    parser.add_argument('--img_ext', type=str, default='.jpg')


    args = parser.parse_args()
    return args

class PrepareFilenames():
    """
    Load two directories filenames to df
    Extracts filenames for metadata folder
    - renames filenams
    - writes filename.txt to metadata folder
    """
    def __init__(self, metadata_flpth, image_data_flpth, text_data_flpth, **kwargs):

        # File paths to data directories
        self.metadata_flpth = metadata_flpth

        self.image_data_flpth = image_data_flpth
        self.text_data_flpth = text_data_flpth

        self.txt_dir = os.path.basename(self.text_data_flpth)
        self.img_dir = os.path.basename(self.image_data_flpth)

        self.basenameCol = kwargs.setdefault('basename_clmn', "basename")
        self.filepathsCol = kwargs.setdefault('filepaths_clmn', "filepaths")

        # 1. Load file names to df from input directories (with explit extentions)
        txt_ext = kwargs.setdefault('txt_ext', ".txt")
        img_ext = kwargs.setdefault('img_ext', ".jpg")
        self.fileNames_df = write.filenames_to_df(self.image_data_flpth, self.text_data_flpth, txt_ext=txt_ext, img_ext=img_ext)

        # 2. Extract filenames for manipulation, and to write the pickle filenames. Returns a long df
        self.fileNames_df[self.basenameCol] = self.extract_filenames(flpthColNms_lst=[self.img_dir, self.txt_dir])


    def extract_filenames(self, flpthColNms_lst=['images','text']):
        """Extracts the base filenames from the txt and image filepaths to a pandas series"""

        # Stack the text and column images flpth to one column
        flpths_df = self.fileNames_df[flpthColNms_lst].stack()
        #print(type(flpths_df))

        # extract the basenames for each of the directories
        baseNm_df = flpths_df.apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        baseNm_df = baseNm_df.unstack()

        try:
            # returns a unified basesname column.  Merge to one column if the basenames are the same
            baseNm_df = baseNm_df.apply(lambda x: x.unique()[0] if x.nunique()==1 else 1/0, axis=1)
        except:
            logging.debug(f"Filenames for images and text do mot match")
            logging.debug(f"Normalize image and text filenames")

            baseNm_df = baseNm_df.stack()
            baseNm_df= self.normalize_basenames(baseNm_df, lowercase=True, strip=True, replace_blanks=True,  _enumerate=False)

            baseNm_df = baseNm_df.unstack()
            baseNm_df = baseNm_df.apply(lambda x: x.unique()[0] if x.nunique()==1 else 1/0, axis=1)

        return baseNm_df


    def normalize_basenames(self, data_series, lowercase=True, strip=True, replace_blanks=True,  _enumerate=False):
        """
        Applies textacy text preprocess to each filename with various options.  Enumerates filenames if enumerate_ is true

        returns an updated fileNames_df

        """
        if strip:
            data_series = data_series.str.strip()

        if replace_blanks:
            data_series =data_series.str.replace(" ","_")

        if lowercase:
            data_series = data_series.str.lower()

        if _enumerate:
            data_series = data_series.str.cat(data_series.index.values.astype(str), sep="_")

        logging.info(f"Finished renaming basenames")

        return data_series

    def rename_filenames(self):
        """
        Write filenames back to disk with new filenames taken from the filename_df basenameCol
        """
        # 1. Reshape to filenames to long form so all the original filepaths are in a list as opposed to tw columns
        filename_df  = pd.melt(self.fileNames_df, id_vars=[self.basenameCol], var_name='filetype', value_name=self.filepathsCol)

        def rename_file(row):
            #print(row)

            # Make new filepath replacing just the filename path
            dir_flpth = os.path.dirname(row[self.filepathsCol])
            filename, file_extension = os.path.splitext(row[self.filepathsCol])
            new_filepath = os.path.join(dir_flpth, f"{row[self.basenameCol]}{ file_extension}")

            os.rename(row[self.filepathsCol], new_filepath)

        # 2. Write the filenames back to disk with stacked df
        filename_df.apply(lambda row: rename_file(row), axis=1)
        logging.info(f"Finished writing new image and text filenames to disk")

        # 3. reset all the columns text and image source filenames to match the filenames on disk
        self.fileNames_df[self.basenameCol] = self.extract_filenames()
        return

    def basenames_to_txtfile(self, basename_flname='filenames.txt'):
        """
        Write filenamee basenames to a txt file
        """
        write_filename_path = os.path.join(self.metadata_flpth, basename_flname)

        self.fileNames_df.to_csv(write_filename_path,
                                  columns=[self.basenameCol],
                                  index=False,
                                  header=False
                                  )

        logging.debug(f"Finished writing filenames to: {write_filename_path} Number of Basenames: {self.fileNames_df[self.basenameCol].shape}")

        return