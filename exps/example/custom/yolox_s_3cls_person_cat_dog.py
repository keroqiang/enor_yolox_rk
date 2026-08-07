#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

from yolox.exp import Exp as MyExp
from yolox.data.datasets.datasets_wrapper import ConcatDataset
from yolox.data.datasets.mapped_coco_dataset import MappedCOCODataset


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.test_conf = 0.5

        # COCO数据集: person + cat + dog
        self.coco_data_dir = None
        self.coco_train_ann = "coco_train_person240k_cat_dog.json"
        self.coco_val_ann = "instances_val2017_area_greater_100.json"
        self.coco_selected_cats = ['person', 'cat', 'dog']

        # Object365 cat/dog
        self.object365_data_dir = "datasets/object365_cat_dog"
        self.object365_train_ann = "train_cat_dog.json"
        self.object365_val_ann = "val_cat_dog.json"
        self.object365_selected_cats = ['cat', 'dog']

        # Oxford Pets cat/dog
        self.pets_data_dir = "datasets/oxford_pets"
        self.pets_train_ann = "train.json"
        self.pets_val_ann = "val.json"
        self.pets_selected_cats = ['cat', 'dog']

        self.class_mapping = {
            'person': 0,
            'cat': 1,
            'dog': 2,
        }
        self.class_names = ['person', 'cat', 'dog']

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1
        self.save_history_ckpt = False
        self.num_classes = len(self.class_names)

    def get_dataset(self, cache: bool = False, cache_type: str = "ram", selected_cat_names=None):
        from yolox.data import TrainTransform

        transform = TrainTransform(
            max_labels=50,
            flip_prob=self.flip_prob,
            hsv_prob=self.hsv_prob
        )

        coco_dataset = MappedCOCODataset(
            data_dir=self.coco_data_dir,
            json_file=self.coco_train_ann,
            img_size=self.input_size,
            preproc=transform,
            cache=cache,
            cache_type=cache_type,
            selected_cat_names=self.coco_selected_cats,
            class_mapping=self.class_mapping
        )

        object365_dataset = MappedCOCODataset(
            data_dir=self.object365_data_dir,
            json_file=self.object365_train_ann,
            name="train",
            img_size=self.input_size,
            preproc=transform,
            cache=cache,
            cache_type=cache_type,
            selected_cat_names=self.object365_selected_cats,
            class_mapping=self.class_mapping
        )

        pets_dataset = MappedCOCODataset(
            data_dir=self.pets_data_dir,
            json_file=self.pets_train_ann,
            name="images",
            img_size=self.input_size,
            preproc=transform,
            cache=cache,
            cache_type=cache_type,
            selected_cat_names=self.pets_selected_cats,
            class_mapping=self.class_mapping
        )

        concat_dataset = ConcatDataset([coco_dataset, object365_dataset, pets_dataset])
        concat_dataset.class_names = self.class_names

        return concat_dataset

    def get_eval_dataset(self, **kwargs):
        from yolox.data import ValTransform

        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        coco_val_dataset = MappedCOCODataset(
            data_dir=self.coco_data_dir,
            json_file=self.coco_val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            selected_cat_names=self.coco_selected_cats,
            class_mapping=self.class_mapping
        )

        object365_val_dataset = MappedCOCODataset(
            data_dir=self.object365_data_dir,
            json_file=self.object365_val_ann,
            name="val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            selected_cat_names=self.object365_selected_cats,
            class_mapping=self.class_mapping
        )

        pets_val_dataset = MappedCOCODataset(
            data_dir=self.pets_data_dir,
            json_file=self.pets_val_ann,
            name="images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            selected_cat_names=self.pets_selected_cats,
            class_mapping=self.class_mapping
        )

        concat_val_dataset = ConcatDataset([coco_val_dataset, object365_val_dataset, pets_val_dataset])
        concat_val_dataset.class_names = self.class_names
        concat_val_dataset.class_ids = list(range(self.num_classes))

        return concat_val_dataset

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import MixedDatasetEvaluator

        return MixedDatasetEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                           testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
