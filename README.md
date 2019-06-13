## pytig

pytig is a python library to automatically generate images from text with a few commands using open source TIG's (text to image generation) algorithms.

    import pytig as tig

    tig.download.datasets(name=['bird'])
    Bird = tig.generate("photosythnesis is a process", model='bird' )
    Bird.img.show()

   ![alt text](example_bird.PNG)

Traditionally TIG's are often developed by researchers, hoping to be published.  Once a paper is published, the algorithms are no longer maintained. Making it difficult for hobbyists, other researchers and engineers to replicate, experiment, and implement in production.

## Things you need to know before you install
1. Greater tham 4 GB of disk space for all the training data and various models.
2. python 3.6+
3. Time and patience.  It all depends on your machine but in general things run in minutes and hours not seconds.





TThe library facilates generating images fro text  using text to image generation algorithms (TIG's) developed by resea






Programmatically generating images from text is a powerful concept with many practical applications in numerous fields such as education, engineering, game making. However in practice using these algorithms to generate images can be difficult i.e. [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf). The primary reasons are (1) Reseaserchers often write the code for the lab algorithms are written by researchers

## Usage
`pip install pytig`

    import pytig as tig

    #test
    # No GPU required
    img = tig.generate("photosythnesis is a process", model='test' )

    #bird
    #coco




### What pytig does.
- Accept text as an input and return an image based on a tig maintained library with three inputs.
    (1) predict text
    (2) algorithm
    (3) Process type i.e. CPU or GPU
- Automate and Log model runs of tig maintained models.
- Help in preparing training data for new models.  A model is a trained object that when fed text can generate an image for that text.
        #### Example 1:
        A model trained on the cub data set (birds)
- Share amd reproduce algorithms

### What tig does not do.
- Supply the compute resources.


## Major Dependancies
pytig is dependant on large packages for preprocessing text, training and predicting images i.e. guild, spacy, textacy, pandas, pytorch

### Terminology
- algorithm - A set of rules and processes which output a model
- model - The algorithm's output which is used to generate the image

- inputs:
    - your train data or use premade example datasets
        -   Example:
        -   Example:
    - sentence(s) you wish to generate images for
        -   Example:
- output:
    -  Image(s)
        -   Example:





# Current Use Cases
- Generate random images of birds:
-
- Generate random images of Live objects
-

# Limitations
- Training Models on your own take a long time
- Lots of Resources
- Current pretrained are limited



# TIG Maintainad Algorithms
### AttnGAN

Pytorch implementation for reproducing AttnGAN results in the paper

**Data**

1. Download our preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`
