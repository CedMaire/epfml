# EPFML - Road Segmentation

## Folders
* `data/` should contain the train and test data (should be at the same level as `src/`)
* `src/` contains all the source files

## How To
1) Download the data set from: ???????????
2) Extract the archive it should create the `data/` folder with following structure: `data/`, `data/???`, ...
3) Run `train.py` from the `root` folder to train the model and predict the test data set
4) Run `run.py` from the `root` folder to predict with the pre-trained model

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
