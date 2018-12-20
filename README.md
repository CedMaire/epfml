# EPFML - Road Segmentation

## Folders
* `data/` should contain the train and test data (should be at the same level as `src/`)
* `src/` contains all the source files

## How To
1) Download the data set from https://drive.google.com/open?id=1FJ05tqUF8VPPqDE923P1wujqpQp9Agws or generate it as explained in the next section
2) Extract the archive to the `root` folder (same level as `src/`)  it should create the `data/` folder
3) Install the following libraries (if not installed already) using `pip install { keras , matplotlib , numpy , opencv-python , tensorflow , sklearn }`
4) Run `python src/train.py` to train the model and predict the test data set
5) Run `python src/run.py` to predict with the pre-trained model

## How To Generate the Complete Dataset
1) Download the 2 data sets from https://www.crowdai.org/challenges/epfl-ml-road-segmentation/dataset_files
2) Extract them to a `data/` folder in the root directory
3) Restructure the `test_set_images/` to obtain the same folder structure presented in the following tree
```
.
├── test_set_images
│   ├── groundtruth
│   └── images
│       ├── test_1
│       ├── test_2
│       ├── ...
│       └── test_50
└── training
    ├── groundtruth
    └── images
```
4) Run `python src/image_multiplier.py` 
5) The files are saved on disk in `data/training/`

## Files
* `train.py` run this to train a new model, this can take several hours
* `run.py` run this to create the CSV that produce our best result in CrowdAi
* `cnn_helper.py` this file contains some method to create our CNN
* `data_loader_saver_helper.py` this file contains helper methods to load ans save images
* `mask_to_submission.py` this file contains the methods to convert images to CSV
* `post_processing_helper.py` this file contains our post-processing methods
* `image_multiplier.py` this file contains the method used to create the additional data

## CrowdId
bdeleze 24966

## Authors
Antonio Morais - antonio.morais@epfl.ch

Benjamin Délèze - benjamin.deleze@epfl.ch

Cedric Maire - cedric.maire@epfl.ch
