#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import numpy as np
from yolox.data import COCODataset


class MappedCOCODataset(COCODataset):
    """
    带类别映射功能的COCO数据集包装类
    用于将不同数据集的原始类别ID映射到统一的类别ID空间
    """
    def __init__(self, *args, class_mapping=None, **kwargs):
        """
        初始化带类别映射的COCO数据集
        
        Args:
            *args: 传递给COCODataset的位置参数
            class_mapping (dict): 类别名称到目标ID的映射字典
            **kwargs: 传递给COCODataset的关键字参数
        """
        # 先初始化必要的属性，确保即使父类初始化失败也能访问
        self.class_mapping = class_mapping or {}
        self.cat_id_to_custom_id = {}
        self.cache = None  # 初始化cache属性，避免__del__错误
        
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 如果设置了selected_cat_names，确保它们都在class_mapping中有映射
        if hasattr(self, 'selected_cat_names') and self.selected_cat_names:
            for cat_name in self.selected_cat_names:
                if cat_name not in self.class_mapping:
                    raise ValueError(f"类别'{cat_name}'在class_mapping中没有映射")
        
        # 构建类别ID映射表
        self._build_custom_cat_id_mapping()
    
    def _build_custom_cat_id_mapping(self):
        """
        构建从原始数据集类别ID到目标统一ID的映射表
        修复：处理同名不同ID的类别情况，确保所有同名类别都能正确映射
        """
        self.cat_id_to_custom_id = {}
        
        # 如果使用了selected_cat_names，只映射选定的类别
        if hasattr(self, 'selected_cat_names') and self.selected_cat_names:
            # 遍历选定的类别
            for cat_name in self.selected_cat_names:
                # 查找原始COCO数据集中所有同名的类别ID
                # 注意：这里移除了break，以处理同名不同ID的类别
                for cat_info in self.coco.cats.values():
                    if cat_info['name'] == cat_name:
                        # 映射到目标ID
                        if cat_name in self.class_mapping:
                            self.cat_id_to_custom_id[cat_info['id']] = self.class_mapping[cat_name]
        else:
            # 映射所有可用类别
            for cat_info in self.coco.cats.values():
                cat_name = cat_info['name']
                if cat_name in self.class_mapping:
                    self.cat_id_to_custom_id[cat_info['id']] = self.class_mapping[cat_name]
    
    def load_anno_from_ids(self, id_):
        """
        重写加载标注的方法，将原始类别ID映射到自定义ID
        注意：当selected_cat_names不为None时，父类返回的res[i][4]已经是相对于selected_cat_names的索引
        """
        # 调用父类方法加载原始标注
        res, img_info, resized_info, file_name = super().load_anno_from_ids(id_)
        
        # 如果有标注，转换类别ID
        if len(res) > 0 and len(res[0]) > 4:  # res格式: [x1, y1, x2, y2, category_id, ...]
            # 遍历所有标注
            for i in range(len(res)):
                # 获取相对于selected_cat_names的索引
                cls_idx = int(res[i][4])
                
                # 如果有selected_cat_names，使用它来获取类别名称
                if hasattr(self, 'selected_cat_names') and self.selected_cat_names and cls_idx < len(self.selected_cat_names):
                    original_cat_name = self.selected_cat_names[cls_idx]
                else:
                    # 否则尝试从原始类别映射中获取
                    original_cat_name = None
                    for cat_id, custom_id in self.cat_id_to_custom_id.items():
                        if custom_id == cls_idx:
                            # 找到对应的原始cat_info
                            for cat_info in self.coco.cats.values():
                                if cat_info['id'] == cat_id:
                                    original_cat_name = cat_info['name']
                                    break
                            break
                
                # 根据类别名称映射到统一的ID
                if original_cat_name in self.class_mapping:
                    res[i][4] = self.class_mapping[original_cat_name]
                else:
                    # 如果没有映射，将其过滤掉
                    res[i] = None
            
            # 过滤掉None值
            res = [r for r in res if r is not None]
        
        return res, img_info, resized_info, file_name
    
    def pull_item(self, index):
        """
        重写pull_item方法，确保返回的标注使用正确的类别ID
        注意：我们不在这里重复类别映射，因为load_anno_from_ids已经处理了映射
        确保返回的label是numpy数组格式，以兼容TrainTransform
        """
        # 调用父类的pull_item方法，正确解包返回值：img, label, origin_image_size, img_id
        img, label, origin_image_size, img_id = super().pull_item(index)
        
        # 严格的类型检查，避免'numpy.int64' has no len()错误
        if label is not None:
            # 确保label是列表或数组类型
            if isinstance(label, (list, np.ndarray)):
                # 只在label不为空且第一个元素也是列表/数组时进行过滤
                if len(label) > 0 and isinstance(label[0], (list, np.ndarray)) and len(label[0]) > 4:
                    # 过滤掉None值
                    label = [l for l in label if l is not None]
                
                # 确保label是numpy数组格式，以兼容TrainTransform类的切片操作
                if isinstance(label, list):
                    label = np.array(label, dtype=np.float32)
        
        # 保持与父类相同的返回顺序
        return img, label, origin_image_size, img_id
