#!/usr/bin/env python3
"""
将 Oxford-IIIT Pets 数据集转换为 COCO 格式
用法: python tools/convert_oxford_pets_to_coco.py
"""

import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict


def convert():
    base_dir = "datasets/oxford_pets"
    xml_dir = os.path.join(base_dir, "annotations", "xmls")
    list_file = os.path.join(base_dir, "annotations", "list.txt")
    img_dir = os.path.join(base_dir, "images")

    # 读取 species 信息
    species_map = {}
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            image_name = parts[0]
            species = int(parts[2])  # 1=cat, 2=dog
            species_map[image_name] = "cat" if species == 1 else "dog"

    # 收集所有有 XML 标注的图片，按 8:2 分 train/val
    import random
    random.seed(42)
    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith(".xml")])
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * 0.8)
    train_xmls = set(xml_files[:split_idx])
    val_xmls = set(xml_files[split_idx:])

    categories = [
        {"id": 1, "name": "cat"},
        {"id": 2, "name": "dog"},
    ]
    cat_name_to_id = {"cat": 1, "dog": 2}

    splits = {
        "train": train_xmls,
        "val": val_xmls,
    }

    for split_name, split_xmls in splits.items():
        images = []
        annotations = []
        ann_id = 1

        for xml_file in sorted(split_xmls):
            image_name = xml_file.replace(".xml", "")

            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()

            filename = root.find("filename").text
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            # 检查图片是否存在
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                # 可能是 jpg 或 JPG
                for ext in [".jpg", ".JPG", ".png", ".PNG"]:
                    alt_path = os.path.join(img_dir, image_name + ext)
                    if os.path.exists(alt_path):
                        filename = image_name + ext
                        break

            img_id = len(images) + 1
            images.append({
                "id": img_id,
                "file_name": filename,
                "width": width,
                "height": height,
            })

            species = species_map.get(image_name, None)

            for obj in root.findall("object"):
                obj_name = obj.find("name").text.lower()
                # 用 species 信息确认 cat/dog
                if species:
                    obj_name = species

                if obj_name not in cat_name_to_id:
                    continue

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                w = xmax - xmin
                h = ymax - ymin
                if w <= 0 or h <= 0:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_name_to_id[obj_name],
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        out_path = os.path.join(base_dir, "annotations", f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        cat_count = defaultdict(int)
        for ann in annotations:
            cat_count[ann["category_id"]] += 1
        print(f"{split_name}: {len(images)} images, {len(annotations)} annotations")
        for cat_id, count in sorted(cat_count.items()):
            name = "cat" if cat_id == 1 else "dog"
            print(f"  {name}: {count}")


if __name__ == "__main__":
    convert()
