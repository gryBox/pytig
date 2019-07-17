## pytig

pytig is a python library to prepare inputs for text to image generator algorithms.  Specifically for the [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) algorithm. (Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He)

### pytig use cases
1. Prepare novel datasets and metadata folder required by the AttnGAN algorithm.
2. Prepare filenames and explore captions i.e. text associated with the images


### AttnGAN Implementations
1. `python 2.7` The authors implementation [AttnGAN algorithm](https://github.com/taoxugit/AttnGAN)
2. `python 3.6` Fork by [davidstap](https://github.com/davidstap/AttnGAN)
3. `python 3.6` Fork by [gryBox](https://github.com/gryBox/AttnGAN). Can be run with or without`guild`.
###


## Usage
`pip install pytig`

    import pytig as tig

    # Prepare MetaData folder for attngan model
    # Inputs
    data_dir_name = 'data'

    metadata_folder_name = 'photosynthesis_raw'
    txt_dir_flname =  "text"
    img_dir_flname =  'images

    data_dir_flpth = os.path.abspath(data_dir_name)
    print(data_dir_flpth)



    # 1.  Prepare Filenames
    # Normalize Names
    prpFilenames = ptg.filenames.PrepareFilenames(
                                                  metadata_flpth,
                                                  image_data_flpth,
                                                  text_data_flpth,
                                                  )

    # Write new filenames back to disk using the new filenames and updates the fileNames_df
    prpFilenames.rename_filenames()

    # Write basenames to a ".txt" file in the metadata folder
    prpFilenames.basenames_to_txtfile(basename_flname='filenames.txt')


**Data**

1. Download our preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`
