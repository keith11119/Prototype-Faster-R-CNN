# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Thursday, April 14, 2022

@author: Guangxing Han
"""
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import pandas as pd
from detectron2.data.catalog import MetadataCatalog

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithSupportBDD"]


class DatasetMapperWithSupportBDD:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        # TODO:  may need to include more transformation to provide a more diverse support set
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        self.few_shot       = cfg.INPUT.FS.FEW_SHOT
        self.support_way    = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot   = cfg.INPUT.FS.SUPPORT_SHOT
        self.use_support_shot_ratio = cfg.INPUT.FS.USE_SUPPORT_SHOT_RATIO
        self.support_shot_ratio = cfg.INPUT.FS.SUPPORT_SHOT_RATIO
        self.seeds          = cfg.DATASETS.SEEDS
        self.support_exclude_query = cfg.INPUT.FS.SUPPORT_EXCLUDE_QUERY
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        if self.is_train:
            # support_df
            self.support_on = True
            if self.few_shot:
                if self.seeds == 0:
                    self.support_df = pd.read_pickle("./datasets/bdd/{}.pkl".format(cfg.DATASETS.TRAIN[0]))
                    print("training support_df=./datasets/bdd/{}.pkl".format(cfg.DATASETS.TRAIN[0]))
                else:
                    self.support_df = pd.read_pickle("./datasets/bdd/seed{}/{}.pkl".format(self.seeds, cfg.DATASETS.TRAIN[0]))
                    print("training support_df=", "./datasets/bdd/seed{}/{}.pkl".format(self.seeds, cfg.DATASETS.TRAIN[0]))
            else:
                for idx, dataset_name in enumerate(cfg.DATASETS.TRAIN):
                    # print('loading support of ', dataset_name)
                    support_df_tmp = pd.read_pickle("./datasets/bdd/{}.pkl".format(dataset_name))
                    print("training support_df=", "./datasets/bdd/{}.pkl".format(dataset_name))
                    if idx == 0:
                        self.support_df = support_df_tmp
                    else:
                        start_id = self.support_df.shape[0]
                        for idx in range(support_df_tmp.shape[0]):
                            support_df_tmp.at[idx, 'id'] += start_id
                        self.support_df = self.support_df.append(support_df_tmp, ignore_index=True)
            print("self.support_df=", self.support_df)
            print("self.support_df.shape=", self.support_df.shape)


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        if self.is_train:
            # support
            if self.support_on:
                if "annotations" in dataset_dict:
                    # USER: Modify this if you want to keep them for some reason.
                    for anno in dataset_dict["annotations"]:
                        if not self.mask_on:
                            anno.pop("segmentation", None)
                        if not self.keypoint_on:
                            anno.pop("keypoints", None)
                support_images, support_bboxes, support_cls = self.generate_support(dataset_dict)
                dataset_dict['support_images'] = torch.as_tensor(np.ascontiguousarray(support_images))
                dataset_dict['support_bboxes'] = support_bboxes
                dataset_dict['support_cls'] = support_cls

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                self.proposal_min_box_size,
                self.proposal_topk,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)
            
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

    def generate_support(self, dataset_dict):
        # not using the support_way from config file any more
        # support_way = self.support_way
        use_support_shot_ratio = self.use_support_shot_ratio

        if use_support_shot_ratio:
            support_shot_ratio = self.support_shot_ratio
        else:
            support_shot = self.support_shot #5
        #support_shot_ratio = self.support_shot_ratio
        
        query_cls = dataset_dict['annotations'][0]['category_id']
        query_img = dataset_dict['image_id']
        all_cls = self.support_df.loc[self.support_df['image_id']==query_img, 'category_id'].tolist()
        # print("all_cls=", all_cls)
        # all_cls = dataset_dict['all_cls']
        all_unique_cls = self.support_df['category_id'].unique()

        support_way = len(all_unique_cls)

        # Crop support data and get new support box in the support data
        support_data_all = []
        support_box_all = []
        if self.support_exclude_query:
            used_image_id = [query_img]
        else:
            used_image_id = []

        used_id_ls = []
        support_category_id = []

        for way, current_cls in enumerate(all_unique_cls):
            if use_support_shot_ratio:
                total_samples_cls = len(self.support_df[self.support_df['category_id'] == current_cls])
                support_shot_cls = max(2, int(total_samples_cls * support_shot_ratio))
            else:
                support_shot_cls = support_shot

            for shot in range(support_shot_cls):
                # Support image and box
                support_list = self.support_df.loc[(self.support_df['category_id'] == current_cls) & (~self.support_df['image_id'].isin(used_image_id)) & (~self.support_df['id'].isin(used_id_ls)), 'id']
                if support_list.empty:
                    continue
                else:
                    support_id = support_list.sample().tolist()[0]
                support_cls = self.support_df.loc[self.support_df['id'] == support_id, 'category_id'].tolist()[0]
                support_img = self.support_df.loc[self.support_df['id'] == support_id, 'image_id'].tolist()[0]
                used_id_ls.append(support_id)
                # used_image_id.append(support_img)

                support_db = self.support_df.loc[self.support_df['id'] == support_id, :]
                assert support_db['id'].values[0] == support_id

                support_data = utils.read_image("./datasets/bdd/" + support_db["file_path"].tolist()[0], format=self.img_format)
                support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                support_box = support_db['support_box'].tolist()[0]
                support_data_all.append(support_data)
                support_box_all.append(support_box)
                support_category_id.append(support_cls)
            print("current_cls=", current_cls)
            print("shot=",support_shot_cls)
            print("support_data_size=", support_data.shape)
            print("support_shot_cls=", support_shot_cls)
            print("support_shot_cls=", total_samples_cls)
        support_data_all = np.stack(support_data_all)
        support_box_all = np.stack(support_box_all)
        
        return support_data_all, support_box_all, support_category_id
