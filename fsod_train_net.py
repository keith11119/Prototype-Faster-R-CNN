#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Created on Thursday, April 14, 2022

This script is a simplified version of the training script in detectron2/tools.

@author: Guangxing Han
"""

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import build_batch_data_loader
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from meta_faster_rcnn.config import get_cfg
from meta_faster_rcnn.data import DatasetMapperWithSupportCOCO, DatasetMapperWithSupportVOC, DatasetMapperWithSupportBDD
from meta_faster_rcnn.data.build import build_detection_train_loader, build_detection_test_loader
from meta_faster_rcnn.solver import build_optimizer
from meta_faster_rcnn.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator, BDDDetectionEvaluator

import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger

class Trainer(DefaultTrainer):

    def run_step(self):

        assert self.model.training, " eval mode to the trainer model"

        data = next(self._data_loader_iter)

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        # Gradient accumulation
        if (self.iter + 1) % self.cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS == 0:
            self.optimizer.step()

        self._write_metrics(loss_dict)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if 'coco' in cfg.DATASETS.TRAIN[0]:
            mapper = DatasetMapperWithSupportCOCO(cfg)
        elif 'voc' in cfg.DATASETS.TRAIN[0]:
            mapper = DatasetMapperWithSupportVOC(cfg)
        elif 'bdd' in cfg.DATASETS.TRAIN[0]:
            mapper = DatasetMapperWithSupportBDD(cfg)
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'coco' in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif 'voc' in dataset_name:
            return PascalVOCDetectionEvaluator(dataset_name)
        elif 'bdd' in dataset_name:
            return BDDDetectionEvaluator(dataset_name)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i

            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Register the GRADIENT_ACCUMULATION_STEPS key
    cfg.SOLVER.register_key("GRADIENT_ACCUMULATION_STEPS")
    cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1

    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="meta_faster_rcnn")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
