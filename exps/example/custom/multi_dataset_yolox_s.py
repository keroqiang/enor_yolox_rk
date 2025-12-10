#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp
from yolox.data.datasets.datasets_wrapper import ConcatDataset
from yolox.data.datasets.mapped_coco_dataset import MappedCOCODataset


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # 模型配置
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # COCO数据集配置
        self.coco_data_dir = None  # 使用默认的datasets目录
        self.coco_train_ann = "instances_train2017.json"
        self.coco_val_ann = "instances_val2017.json"
        self.coco_selected_cats = ['person', 'cat', 'dog', 'chair', 'couch']
        
        # Charger数据集配置
        self.charger_data_dir = "datasets/charger"
        self.charger_train_ann = "train_annotations.coco.json"
        self.charger_val_ann = "val_annotations.coco.json"
        self.charger_selected_cats = ['charger']
        
        # 类别映射：将不同数据集的类别ID映射到统一的ID
        self.class_mapping = {
            # COCO数据集类别映射
            'person': 0,
            'cat': 1,
            'dog': 2,
            'chair': 3,
            'couch': 4,
            # Charger数据集类别映射
            'charger': 5
        }
        
        # 类别名称顺序，与class_mapping中的ID对应
        self.class_names = ['person', 'cat', 'dog', 'chair', 'couch', 'charger']
        
        self.max_epoch = 200
        self.data_num_workers = 8
        self.eval_interval = 1
        self.save_history_ckpt = False
        self.num_classes = len(self.class_names)

    def get_dataset(self, cache: bool = False, cache_type: str = "ram", selected_cat_names=None):
        """
        重写get_dataset方法，使用ConcatDataset合并多个数据集
        注意：selected_cat_names参数在此实现中被忽略，因为我们已在方法中硬编码了类别选择
        """
        from yolox.data import TrainTransform
        
        # 创建COCO数据集实例，只选择猫和狗类别，并使用类别映射
        coco_dataset = MappedCOCODataset(
            data_dir=self.coco_data_dir,
            json_file=self.coco_train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
            selected_cat_names=self.coco_selected_cats,
            class_mapping=self.class_mapping
        )
        
        # 创建Charger数据集实例，并使用类别映射
        charger_dataset = MappedCOCODataset(
            data_dir=self.charger_data_dir,
            json_file=self.charger_train_ann,
            name="train",  # 指定charger数据集的图像目录名为"train"
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
            selected_cat_names=self.charger_selected_cats,
            class_mapping=self.class_mapping
        )
        
        # 使用ConcatDataset合并两个数据集
        concat_dataset = ConcatDataset([coco_dataset, charger_dataset])
        
        # 确保合并后的数据集知道类别名称
        concat_dataset.class_names = self.class_names
        
        return concat_dataset

    def get_eval_dataset(self, **kwargs):
        """
        重写get_eval_dataset方法，支持混合数据集的评估
        """
        from yolox.data import ValTransform
        
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        
        # 创建评估用的COCO数据集，只选择猫和狗类别，并使用类别映射
        coco_val_dataset = MappedCOCODataset(
            data_dir=self.coco_data_dir,
            json_file=self.coco_val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            selected_cat_names=self.coco_selected_cats,
            class_mapping=self.class_mapping
        )
        
        # 创建评估用的Charger数据集，并使用类别映射
        charger_val_dataset = MappedCOCODataset(
            data_dir=self.charger_data_dir,
            json_file=self.charger_val_ann,
            name="val",  # 指定charger验证数据集的图像目录名为"val"
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            selected_cat_names=self.charger_selected_cats,
            class_mapping=self.class_mapping
        )
        
        # 使用ConcatDataset合并评估数据集
        concat_val_dataset = ConcatDataset([coco_val_dataset, charger_val_dataset])
        
        # 确保合并后的评估数据集知道类别名称和类别ID
        concat_val_dataset.class_names = self.class_names
        # 添加class_ids属性，用于COCOEvaluator
        concat_val_dataset.class_ids = list(range(self.num_classes))
        # 添加coco属性，使用第一个数据集的coco对象，适应原项目的COCOEvaluator;目前使用的是MixedDatasetEvaluator，和COCOEvaluator无关，先注释掉
        # concat_val_dataset.coco = coco_val_dataset.coco
        
        return concat_val_dataset

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """
        重写get_evaluator方法，使用正确的类别数量
        """
        from yolox.evaluators import MixedDatasetEvaluator
        
        return MixedDatasetEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                           testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,  # 使用混合数据集的类别数量
            testdev=testdev,
        )
