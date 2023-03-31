# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

@moodified from Guangxing Han's work
"""

# BDD categories
#need to give the new categories
BDD_ALL_CATEGORIES = {
    # 1: ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
    #     'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
    #     'train', 'tvmonitor', 'bird', 'bus', 'cow', 'motorbike', 'sofa'],
    # 2: ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
    #     'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
    #     'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    # 3: ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
    #     'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
    #     'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa'],

    1: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle', 'traffic light', 'traffic sign'],
}

BDD_NOVEL_CATEGORIES = {
    1: ['bird', 'bus', 'cow', 'motorbike', 'sofa'],
    2: ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    3: ['boat', 'cat', 'motorbike', 'sheep', 'sofa'],
}

BDD_BASE_CATEGORIES = {
    1: ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor'],
    2: ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor'],
    3: ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor'],
}

def _get_bdd_fewshot_instances_meta():
    ret = {
        "thing_classes": BDD_ALL_CATEGORIES,
        "novel_classes": BDD_NOVEL_CATEGORIES,
        "base_classes": BDD_BASE_CATEGORIES,
    }
    return ret


def _get_builtin_metadata_bdd(dataset_name):
    if dataset_name == "bdd_fewshot":
        return _get_bdd_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
