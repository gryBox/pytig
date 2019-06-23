import os
import textacy


import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# class PrepareFilenames():
#     """
#     Prepares base file names to match between text and image directories
#     """

def load_filenames(image_dir_path, text_dir_path):
    """
    Loads file names from text and image directory
    """
    # Check that the image files and text files have the same base name and lowercase
    # Load files

    key_names_dict = {
        os.path.basename(text_dir_path): text_dir_path,
        os.path.basename(image_dir_path): image_dir_path
    }

    # Load filenames into a dict split into text and image file lists
    file_names_dict = dict()

    for data_name, filepath in  key_names_dict:

        # Only execute
        if os.path.exists(filepath):
            file_names_dict[data_name] = textacy.io.utils.get_filepaths(
                filepath,
                extension=None,
                ignore_invisible=True,
                recursive=True)
        else:
            logging.info("The dir:{filepath} does not exist")
            assert False






    return file_names_dict