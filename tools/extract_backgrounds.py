#!/usr/bin/env python3
"""
从 COCO 数据集中提取不含指定类别的图片作为背景。

用法：
  python tools/extract_backgrounds.py \
    --coco-json datasets/COCO/annotations/instances_train2017.json \
    --img-dir datasets/COCO/train2017 \
    --output-dir datasets/COCO/backgrounds \
    --exclude person,cat,dog \
    --max-count 5000
"""

import json
import os
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='从COCO提取背景图片')
    parser.add_argument('--coco-json', required=True, help='COCO标注JSON文件')
    parser.add_argument('--img-dir', required=True, help='COCO图片目录')
    parser.add_argument('--output-dir', required=True, help='背景图输出目录')
    parser.add_argument('--exclude', default='person,cat,dog',
                        help='要排除的类别名，逗号分隔（默认: person,cat,dog）')
    parser.add_argument('--max-count', type=int, default=5000,
                        help='最多提取多少张背景图（默认: 5000）')
    args = parser.parse_args()

    # 加载标注
    with open(args.coco_json) as f:
        data = json.load(f)

    # 找出要排除的类别ID
    exclude_names = set(args.exclude.split(','))
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    exclude_cat_ids = {cat_id for cat_id, name in categories.items() if name in exclude_names}
    print(f'排除类别: {exclude_names} -> ID: {exclude_cat_ids}')

    # 找出包含这些类别的图片
    images_with_excluded = set()
    for ann in data['annotations']:
        if ann['category_id'] in exclude_cat_ids:
            images_with_excluded.add(ann['image_id'])

    # 没有排除类别的图片
    all_images = {img['id']: img for img in data['images']}
    bg_image_ids = [img_id for img_id in all_images if img_id not in images_with_excluded]

    print(f'COCO总图片数: {len(all_images)}')
    print(f'包含排除类别的图片: {len(images_with_excluded)}')
    print(f'可用背景图: {len(bg_image_ids)}')

    if not bg_image_ids:
        print('错误: 没有可用的背景图')
        return

    # 随机采样
    import random
    random.seed(42)
    if len(bg_image_ids) > args.max_count:
        bg_image_ids = random.sample(bg_image_ids, args.max_count)
        print(f'随机采样 {args.max_count} 张')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 复制背景图
    copied = 0
    skipped = 0
    for img_id in bg_image_ids:
        img_info = all_images[img_id]
        src_path = os.path.join(args.img_dir, img_info['file_name'])
        dst_path = os.path.join(args.output_dir, img_info['file_name'])

        if not os.path.exists(src_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied += 1

        if copied % 100 == 0:
            print(f'  已复制 {copied} 张...')

    print(f'\n提取完成:')
    print(f'  复制图片: {copied}')
    print(f'  跳过图片: {skipped}')
    print(f'  输出目录: {args.output_dir}')


if __name__ == '__main__':
    main()
