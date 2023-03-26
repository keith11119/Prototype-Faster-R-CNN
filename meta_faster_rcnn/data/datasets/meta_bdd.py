# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Thursday, April 14, 2022

@author: Guangxing Han
"""

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_meta_bdd"]


def load_filtered_bdd_instances(
    name: str, dirname: str, split: str, classnames: str):
    """
    Load BDD VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main",
                                       split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []

    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if not (cls in classnames):
                continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0

            instances.append({
                "category_id": classnames.index(cls),
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
            })
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_meta_bdd(
    name, metadata, dirname, split, keepclasses, sid):
    if keepclasses.startswith('all'):
        thing_classes = metadata["thing_classes"][sid]
    elif keepclasses.startswith('base'):
        thing_classes = metadata["base_classes"][sid]
    elif keepclasses.startswith('novel'):
        thing_classes = metadata["novel_classes"][sid]

    DatasetCatalog.register(
        name, lambda: load_filtered_bdd_instances(
            name, dirname, split, thing_classes)
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes, dirname=dirname, split=split
    )
