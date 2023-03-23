# Prototype-Faster-R-CNN
change of MIN_SIZE_TRAIN is needed
change of STEPS and size of batch are needed 
change of support ways are needed -> should be the number of total classes the previous way to classify into noval and base classes



## Highlights

- Our model is a natural extension of Faster R-CNN for few-shot scenario with the prototype based metric-learning.
- Our meta-learning based models achieve strong few-shot object detection performance without fine-tuning.
- Our model can keep the knowledge of base classes by learning a separate Faster R-CNN detection head for base classes.

## Installation

Our codebase is built upon [detectron2](https://github.com/facebookresearch/detectron2). You only need to install [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) following their instructions.

Please note that we used detectron 0.2.1 in this project. Higher versions of detectron might report errors.

## Data Preparation

- We evaluate our model on two FSOD benchmarks PASCAL VOC and MSCOCO following the previous work [TFA](https://github.com/ucbdrive/few-shot-object-detection).
- Please prepare the original PASCAL VOC and MSCOCO datasets and also the few-shot datasets following [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) in the folder ./datasets/coco and ./datasets/pascal_voc respectively.
- Please run the scripts in ./datasets/coco and ./datasets/pascal_voc step by step to generate the support images for both many-shot base classes (used during meta-training) and few-shot classes (used during few-shot fine-tuning).


## Acknowledgement

This repo is developed based on [Meta-Faster-R-CNN](https://github.com/GuangxingHan/Meta-Faster-R-CNN) and [detectron2](https://github.com/facebookresearch/detectron2). Thanks for their wonderful codebases.
