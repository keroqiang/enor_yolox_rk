#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch
from pycocotools.coco import COCO

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


# 直接从coco_evaluator.py复制的工具函数
def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class MixedDatasetEvaluator:
    """
    混合数据集COCO AP评估类。专为处理多个数据集合并后的评估场景设计，
    支持处理ConcatDataset对象，确保正确评估混合数据集上的模型性能。
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        
        # 获取类别信息
        class_ids = self.dataloader.dataset.class_ids
        class_names = self.dataloader.dataset.class_names
        
        # 获取类别映射信息，如果有的话
        class_mapping = None
        if hasattr(self.dataloader.dataset, 'datasets'):
            # 从第一个子数据集获取类别映射
            for sub_dataset in self.dataloader.dataset.datasets:
                if hasattr(sub_dataset, 'class_mapping'):
                    class_mapping = sub_dataset.class_mapping
                    break
        
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            # 模型输出的类别索引已经是统一的类别ID（0, 1, 2）
            categories = []
            for ind in range(bboxes.shape[0]):
                # 直接使用模型输出的类别索引作为category_id
                category_id = int(cls[ind])
                categories.append(category_id)

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": categories,
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                # 直接使用模型输出的类别索引作为category_id
                # 这个ID已经是通过class_mapping映射后的统一ID
                category_id = int(cls[ind])
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": category_id,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def convert_numpy_types(self, obj):
        """
        将NumPy数据类型转换为Python原生类型，以便JSON序列化
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            # 对于混合数据集，我们需要收集所有子数据集的真实标注数据
            # 创建一个完整的COCO格式ground truth对象
            merged_gt = {
                "images": [],
                "annotations": [],
                "categories": [],
                "info": {"description": "Merged COCO Ground Truth for Mixed Datasets"},
                "licenses": []
            }
            
            # 添加类别信息（使用统一的类别索引）
            for i, class_name in enumerate(self.dataloader.dataset.class_names):
                merged_gt["categories"].append({
                    "id": i,  # 直接使用0, 1, 2作为类别ID
                    "name": class_name,
                    "supercategory": "object"
                })
            
            # 收集所有子数据集的图像和标注信息
            dataset = self.dataloader.dataset
            if hasattr(dataset, 'datasets'):  # 如果是ConcatDataset
                # 遍历所有子数据集
                for sub_dataset in dataset.datasets:
                    if hasattr(sub_dataset, 'coco') and sub_dataset.coco is not None:
                        # 收集图像信息
                        for img_info in sub_dataset.coco.dataset['images']:
                            # 确保图像ID唯一
                            img_id = img_info['id']
                            # 转换NumPy类型
                            converted_img_info = self.convert_numpy_types(img_info)
                            if not any(existing_img['id'] == img_id for existing_img in merged_gt['images']):
                                merged_gt['images'].append(converted_img_info)
                        
                        # 收集标注信息并进行类别映射
                        # 使用子数据集的cat_id_to_custom_id映射表（这是MappedCOCODataset中构建的）
                        if hasattr(sub_dataset, 'cat_id_to_custom_id'):
                            for ann in sub_dataset.coco.dataset['annotations']:
                                # 复制标注信息
                                new_ann = self.convert_numpy_types(ann.copy())
                                
                                # 获取原始类别ID
                                orig_cat_id = ann['category_id']
                                
                                # 使用子数据集的cat_id_to_custom_id映射表进行映射
                                if orig_cat_id in sub_dataset.cat_id_to_custom_id:
                                    new_ann['category_id'] = sub_dataset.cat_id_to_custom_id[orig_cat_id]
                                    # 只添加有效的标注（确保类别ID在0到类别总数-1之间）
                                    if 0 <= new_ann['category_id'] < len(self.dataloader.dataset.class_names):
                                        merged_gt['annotations'].append(new_ann)
                        else:
                            # 如果没有直接的ID映射表，尝试通过类别名称映射
                            for ann in sub_dataset.coco.dataset['annotations']:
                                # 复制标注信息
                                new_ann = self.convert_numpy_types(ann.copy())
                                
                                # 获取原始类别ID
                                orig_cat_id = ann['category_id']
                                
                                # 查找该类别ID对应的名称
                                orig_cat_name = None
                                for cat in sub_dataset.coco.dataset['categories']:
                                    if cat['id'] == orig_cat_id:
                                        orig_cat_name = cat['name']
                                        break
                                
                                # 如果找到类别名称，尝试在统一类别中找到对应的索引
                                if orig_cat_name is not None:
                                    # 在统一的类别名称列表中查找索引
                                    if orig_cat_name in self.dataloader.dataset.class_names:
                                        new_ann['category_id'] = self.dataloader.dataset.class_names.index(orig_cat_name)
                                        merged_gt['annotations'].append(new_ann)
            
            # 转换预测结果中的NumPy类型
            converted_data_dict = self.convert_numpy_types(data_dict)
            
            # 创建临时文件来保存合并后的ground truth
            _, tmp_gt = tempfile.mkstemp()
            json.dump(merged_gt, open(tmp_gt, "w"))
            
            # 加载合并后的ground truth
            cocoGt = COCO(tmp_gt)
            
            # 保存检测结果
            if self.testdev:
                json.dump(converted_data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp_dt = tempfile.mkstemp()
                json.dump(converted_data_dict, open(tmp_dt, "w"))
                cocoDt = cocoGt.loadRes(tmp_dt)
            
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval
                logger.warning("Use standard COCOeval.")

            # 创建COCOeval实例
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            
            # 使用混合数据集的类别ID和类别名称
            cat_ids = self.dataloader.dataset.class_ids
            cat_names = self.dataloader.dataset.class_names
            
            # 设置评估参数
            cocoEval.params.catIds = cat_ids
            cocoEval.params.imgIds = cocoGt.getImgIds()  # 使用所有图像进行评估
            
            # 执行评估
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            
            # 添加类别AP和AR表格
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
