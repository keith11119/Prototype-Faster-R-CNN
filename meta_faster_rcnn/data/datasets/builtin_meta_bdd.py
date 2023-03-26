# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Created on Thursday, April 14, 2022

"""

# BDD categories
BDD_ALL_CATEGORIES = {
    1: ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle', 'traffic light', 'traffic sign'],
}


def _get_bdd_instances_meta():
    ret = {
        "thing_classes": BDD_ALL_CATEGORIES,
    }
    return ret


def _get_builtin_metadata_bdd(dataset_name):
    if dataset_name == "bdd":
        return _get_bdd_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
