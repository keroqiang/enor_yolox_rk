#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODataset(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
        selected_cat_names=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            selected_cat_names (list): list of category names to train on. If None, train on all categories.
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.selected_cat_names = selected_cat_names

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        
        # 过滤类别
        if selected_cat_names is not None:
            # 获取所有类别信息
            all_cats = self.coco.loadCats(self.coco.getCatIds())
            # 创建类别名称到id的映射
            name_to_id = {cat["name"]: cat["id"] for cat in all_cats}
            # 验证选择的类别是否都存在
            for cat_name in selected_cat_names:
                if cat_name not in name_to_id:
                    raise ValueError(f"Category '{cat_name}' not found in COCO dataset")
            # 获取选定类别的id
            self.class_ids = sorted([name_to_id[name] for name in selected_cat_names])
            self.cats = [cat for cat in all_cats if cat["id"] in self.class_ids]
            self._classes = tuple(selected_cat_names)
            # 获取包含选定类别的图像id
            img_ids_with_selected_cats = []
            for cat_id in self.class_ids:
                img_ids_with_selected_cats.extend(self.coco.getImgIds(catIds=[cat_id]))
            # 去重并排序
            self.ids = sorted(list(set(img_ids_with_selected_cats)))
            
            # 打印统计信息，确认类别过滤是否正常工作
            print(f"===== COCO Dataset Category Filtering =====")
            print(f"Selected categories: {selected_cat_names}")
            print(f"Original total images: {len(self.coco.getImgIds())}")
            print(f"Filtered images count: {len(self.ids)}")
            print(f"===== COCO Dataset Category Filtering =====")
        else:
            # 使用所有类别
            self.class_ids = sorted(self.coco.getCatIds())
            self.cats = self.coco.loadCats(self.coco.getCatIds())
            self._classes = tuple([c["name"] for c in self.cats])
            self.ids = self.coco.getImgIds()
        
        self.num_imgs = len(self.ids)
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type,
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        annotations = [self.load_anno_from_ids(_ids) for _ids in self.ids]
        
        # 计算标注数量统计
        total_objects = 0
        class_counts = {i: 0 for i in range(len(self._classes))}
        for anno in annotations:
            # anno[0] 是标注数据 (num_objs, 5)
            objects = anno[0]
            total_objects += len(objects)
            for obj in objects:
                class_idx = int(obj[4])
                if class_idx in class_counts:
                    class_counts[class_idx] += 1
        
        # 打印标注统计信息
        print(f"===== Annotations Statistics =====")
        print(f"Total objects annotated: {total_objects}")
        print(f"Objects per class:")
        for i, class_name in enumerate(self._classes):
            print(f"  - {class_name}: {class_counts.get(i, 0)}")
        print(f"===== Annotations Statistics =====")
        
        return annotations

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        
        # 如果指定了类别过滤，只获取选定类别的标注
        if self.selected_cat_names is not None:
            anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], catIds=self.class_ids, iscrowd=False)
        else:
            anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
            
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            # 确保只处理选定类别的对象（额外检查以防万一）
            if self.selected_cat_names is None or obj["category_id"] in self.class_ids:
                x1 = np.max((0, obj["bbox"][0]))
                y1 = np.max((0, obj["bbox"][1]))
                x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
                y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
                if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                    obj["clean_bbox"] = [x1, y1, x2, y2]
                    objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
