"""
Created on Thursday, April 14, 2022

@author: Guangxing Han
"""
import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .builtin_meta_bdd import _get_builtin_metadata_bdd
from .meta_bdd import register_meta_bdd
from detectron2.data import MetadataCatalog


def register_all_bdd(root="datasets"):
    # register meta datasets
    METASPLITS = [
        ("bdd_trainval_all", "BDD", "trainval", "all", 1),
        ("bdd_train_all", "BDD", "train", "all", 1),
        ("bdd_val_all", "BDD", "val", "all", 1),
        ("bdd_test_all", "BDD", "test", "all", 1)
    ]

    for name, dirname, split, keepclasses, sid in METASPLITS:
        if "BDD" in dirname:
            register_meta_bdd(name,
                             _get_builtin_metadata_bdd("bdd"),
                             os.path.join(root, dirname), split,
                             keepclasses, sid)
            MetadataCatalog.get(name).evaluator_type = "bdd"


# Register them all under "./datasets"

_root = os.getenv("DETECTRON2_DATASETS", "datasets/bdd")
register_all_bdd(_root)
