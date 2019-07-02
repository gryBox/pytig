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
    Load two directories filenames to df
    Extracts filenames for metadata folder
    - renames filenams
    - writes filename.txt to metadata folder
    """
    def __init__(self, metadata_flpth, image_training_data_flpth, text_training_data_flpth, **kwargs):

        # File paths to data directories
        self.metadata_flpth = metadata_flpth

        self.image_training_data_flpth = image_training_data_flpth
        self.text_training_data_flpth = text_training_data_flpth

        self.txt_dir = os.path.basename(self.text_training_data_flpth)
        self.img_dir = os.path.basename(self.image_training_data_flpth)

        self.basenameCol = kwargs.setdefault('filename_clmn','filename')

        # Load file names to df from input directories (with explit extentions)
        txt_ext = kwargs.setdefault('txt_ext', ".txt")
        img_ext = kwargs.setdefault('img_ext', ".jpg")
        self.fileNames_df = write.filenames_to_df(self.image_training_data_flpth, self.text_training_data_flpth, txt_ext=txt_ext, img_ext=img_ext)

        # Extract filenames for manipulation and to write the pickle filenames

        self.fileNames_df[self.basenameCol] = self.extract_filenames(self.fileNames_df)

    def extract_filenames(self, df, flpthColNms_lst=['images','text']):
        """Extracts the base filenames from the txt and image filepaths to a list"""

        # Stack the text and column images flpth to one column
        flpths_df = df[flpthColNms_lst].stack()
        #print(type(flpths_df))

        # extract the basenames for each of the directories
        baseNm_df = flpths_df.apply(lambda x: os.path.splitext(os.path.basename(x))[0]).unstack()

        # # Merge to one column if the basenames are the same
        error = [1,0]
        baseNm_series = baseNm_df.apply(lambda x: x.unique()[0] if x.nunique()==1 else error, axis=1)

        return baseNm_series


    def rename_basename(self, preprocess_text=True, _enumerate=False, **kwargs):
        """
        Applies textacy text preprocess to each filename with various options.  Enumerates filenames if enumerate_ is true

        returns an updated fileNames_df

        """

        if preprocess_text:
            # Apply textacy
            self.fileNames_df[self.basenameCol] = self.fileNames_df[self.basenameCol].apply(lambda x:
                textacy.preprocess.preprocess_text(
                x,
                #normalized_unicode=kwargs.setdefault('normalized_unicode', True), textacy bug
                lowercase=kwargs.setdefault('lowercase', True),
                no_urls=kwargs.setdefault('no_urls', True),
                no_emails=kwargs.setdefault('no_emails', True),
                no_phone_numbers=kwargs.setdefault('no_phone_numbers', True),
                no_numbers=kwargs.setdefault('no_numbers', False),
                no_currency_symbols=kwargs.setdefault('no_currency_symbols', True),
                no_punct=kwargs.setdefault('no_punct', False),
                no_contractions=kwargs.setdefault('no_contractions', True),
                no_accents=kwargs.setdefault('no_accents', True)
                ))

        if _enumerate:
            self.fileNames_df[self.basenameCol] = self.fileNames_df[self.basenameCol].str.cat(self.fileNames_df.index.values.astype(str), sep="_")

        logging.info(f"Finished renaming basenames")

        return self.fileNames_df

    def rename_filenames(self):
        """
        Write filenames back to disk with new filenames taken from the filename_df basenameCol
        """
        # 1. Reshape to filenames to long form so all the original filepaths are in a list as opposed to tw columns
        filename_df  = pd.melt(self.fileNames_df, id_vars=[self.basenameCol], var_name='filetype', value_name='orig_filepath')

        def rename_file(row):
            #print(row)

            # Make new filepath replacing just the filename path
            dir_flpth = os.path.dirname(row['orig_filepath'])
            filename, file_extension = os.path.splitext(row['orig_filepath'])
            new_filepath = os.path.join(dir_flpth, f"{row[self.basenameCol]}{ file_extension}")

            os.rename(row['orig_filepath'], new_filepath)

        # 2. Write the filenames back to disk with stacked df
        filename_df.apply(lambda row: rename_file(row), axis=1)
        logging.info(f"Finished writing new image and text filenames to disk")

        # 3. reset all the columns text and image source filenames to match the filenames on disk
        self.fileNames_df[self.basenameCol] = self.extract_filenames(self.fileNames_df)
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