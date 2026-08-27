#!/usr/bin/env python3
"""
将数据集中的大目标缩放成中小目标，生成新的图片和标注。

两种模式：
  1. letterbox 模式（默认）：保持长宽比缩小，灰色填充
  2. 背景模式：保持长宽比缩小，粘贴到真实背景图上

用法：
  # letterbox 模式
  python tools/shrink_targets.py \
    --input-json datasets/open_images_v7/annotations/train_cat_dog.json \
    --input-img-dir datasets/open_images_v7/train \
    --output-dir datasets/open_images_v7_shrunk \
    --target-size 64

  # 背景模式
  python tools/shrink_targets.py \
    --input-json datasets/open_images_v7/annotations/train_cat_dog.json \
    --input-img-dir datasets/open_images_v7/train \
    --output-dir datasets/open_images_v7_shrunk \
    --target-size 64 \
    --background-dir datasets/COCO/backgrounds
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image
import random


def calculate_shrink_factor(bbox_w, bbox_h, img_w, img_h, target_pixel, img_size=640):
    """
    计算缩放比例，使得目标在 img_size x img_size 的输入中边长约为 target_pixel 像素。
    """
    shrink_w = target_pixel * img_w / (bbox_w * img_size)
    shrink_h = target_pixel * img_h / (bbox_h * img_size)
    return min(shrink_w, shrink_h)


def process_image_letterbox(img_path, annotations, shrink_factor, output_img_path):
    """
    letterbox 模式：保持长宽比缩小，灰色填充居中。
    """
    img = Image.open(img_path)
    orig_w, orig_h = img.size

    r = shrink_factor
    new_w = int(orig_w * r)
    new_h = int(orig_h * r)
    if new_w < 1 or new_h < 1:
        return []

    resized_img = img.resize((new_w, new_h), Image.BILINEAR)

    # 灰色画布，居中放置
    canvas = Image.new('RGB', (orig_w, orig_h), (114, 114, 114))
    paste_x = (orig_w - new_w) // 2
    paste_y = (orig_h - new_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))
    canvas.save(output_img_path)

    # 更新标注
    new_annotations = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        new_x = x * r + paste_x
        new_y = y * r + paste_y
        new_w = w * r
        new_h = h * r

        if new_x + new_w < 0 or new_y + new_h < 0:
            continue
        if new_x >= orig_w or new_y >= orig_h:
            continue

        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, orig_w - new_x)
        new_h = min(new_h, orig_h - new_y)

        if new_w < 1 or new_h < 1:
            continue

        new_ann = ann.copy()
        new_ann['bbox'] = [new_x, new_y, new_w, new_h]
        new_ann['area'] = new_w * new_h
        new_annotations.append(new_ann)

    return new_annotations


def process_image_background(img_path, annotations, shrink_factor, output_img_path,
                             bg_img, bg_w, bg_h):
    """
    背景模式：保持长宽比缩小，粘贴到真实背景图上。
    """
    img = Image.open(img_path)
    orig_w, orig_h = img.size

    r = shrink_factor
    new_w = int(orig_w * r)
    new_h = int(orig_h * r)
    if new_w < 1 or new_h < 1:
        return []

    resized_img = img.resize((new_w, new_h), Image.BILINEAR)

    # 背景图需要 resize 到和原图一样大（或者原图 resize 到背景图大小）
    # 这里选择：背景图 resize 到原图尺寸
    if bg_w != orig_w or bg_h != orig_h:
        bg_img = bg_img.resize((orig_w, orig_h), Image.BILINEAR)

    # 随机放置位置
    paste_x = random.randint(0, max(0, orig_w - new_w))
    paste_y = random.randint(0, max(0, orig_h - new_h))

    # 粘贴到背景上
    bg_img.paste(resized_img, (paste_x, paste_y))
    bg_img.save(output_img_path)

    # 更新标注
    new_annotations = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        new_x = x * r + paste_x
        new_y = y * r + paste_y
        new_w = w * r
        new_h = h * r

        if new_x + new_w < 0 or new_y + new_h < 0:
            continue
        if new_x >= orig_w or new_y >= orig_h:
            continue

        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, orig_w - new_x)
        new_h = min(new_h, orig_h - new_y)

        if new_w < 1 or new_h < 1:
            continue

        new_ann = ann.copy()
        new_ann['bbox'] = [new_x, new_y, new_w, new_h]
        new_ann['area'] = new_w * new_h
        new_annotations.append(new_ann)

    return new_annotations


def load_background_images(bg_dir, max_count=10000):
    """加载背景图片路径列表"""
    bg_paths = []
    for root, dirs, files in os.walk(bg_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                bg_paths.append(os.path.join(root, f))
                if len(bg_paths) >= max_count:
                    return bg_paths
    return bg_paths


def main():
    parser = argparse.ArgumentParser(description='缩放大目标为中小目标')
    parser.add_argument('--input-json', required=True, help='输入COCO标注JSON文件')
    parser.add_argument('--input-img-dir', required=True, help='输入图片目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--target-size', type=int, default=64,
                        help='目标在640x640输入中的边长(像素)，默认64(中目标)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='训练输入尺寸，默认640')
    parser.add_argument('--min-shrink', type=float, default=0.05,
                        help='最小缩放比例，默认0.05')
    parser.add_argument('--max-shrink', type=float, default=1.0,
                        help='最大缩放比例，默认1.0(不缩放)')
    parser.add_argument('--mode', choices=['all', 'large-only'], default='large-only',
                        help='all: 缩放所有目标, large-only: 只缩放大目标')
    parser.add_argument('--background-dir', default=None,
                        help='背景图片目录（不指定则用letterbox灰色填充）')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'],
                        help='数据集划分，默认train')
    args = parser.parse_args()

    # 加载标注
    with open(args.input_json) as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    # 按图片分组标注
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # 加载背景图
    bg_paths = []
    if args.background_dir:
        bg_paths = load_background_images(args.background_dir)
        if not bg_paths:
            print(f'错误: 背景目录 {args.background_dir} 中没有找到图片')
            return
        print(f'加载了 {len(bg_paths)} 张背景图片')

    # 创建输出目录（保持和原数据集一致的结构）
    # 结构: datasets/open_images_v7_shrunk/{split}/images/*.jpg
    output_split_dir = os.path.join(args.output_dir, args.split)
    output_img_dir = os.path.join(output_split_dir, 'images')
    output_ann_dir = os.path.join(args.output_dir, 'annotations')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)

    # 大目标阈值 (在640x640中边长 >= 96)
    large_threshold = 96 * 96  # 9216

    new_images = []
    new_annotations = []
    new_ann_id = 1
    processed = 0
    skipped = 0
    total_anns_original = 0
    total_anns_new = 0

    mode_str = "背景模式" if bg_paths else "letterbox模式"
    print(f'处理模式: {mode_str}')
    print(f'目标大小: {args.target_size}px (在{args.img_size}x{args.img_size}输入中)')

    for img_id, img_info in images.items():
        anns = img_annotations.get(img_id, [])
        if not anns:
            continue

        img_path = os.path.join(args.input_img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img_w = img_info['width']
        img_h = img_info['height']

        # 计算每个目标需要的缩放比例，取最小值
        shrink_factors = []
        for ann in anns:
            bx, by, bw, bh = ann['bbox']
            r = min(args.img_size / img_w, args.img_size / img_h)
            scaled_area = (bw * r) * (bh * r)

            if args.mode == 'large-only' and scaled_area < large_threshold:
                shrink_factors.append(1.0)
            else:
                sf = calculate_shrink_factor(bw, bh, img_w, img_h,
                                             args.target_size, args.img_size)
                sf = max(args.min_shrink, min(args.max_shrink, sf))
                shrink_factors.append(sf)

        if not shrink_factors:
            continue

        shrink_factor = min(shrink_factors)
        if shrink_factor >= 0.95:
            skipped += 1
            continue

        # 处理图片
        # file_name 是 "images/xxx.jpg"，只需要文件名部分
        img_filename = os.path.basename(img_info['file_name'])
        output_img_path = os.path.join(output_img_dir, img_filename)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

        if bg_paths:
            # 背景模式
            bg_path = random.choice(bg_paths)
            bg_img = Image.open(bg_path).copy()
            bg_w, bg_h = bg_img.size
            new_anns = process_image_background(
                img_path, anns, shrink_factor, output_img_path,
                bg_img, bg_w, bg_h
            )
        else:
            # letterbox 模式
            new_anns = process_image_letterbox(
                img_path, anns, shrink_factor, output_img_path
            )

        if not new_anns:
            skipped += 1
            continue

        # 保存新图片信息
        new_img_info = img_info.copy()
        new_img_info['id'] = len(new_images) + 1
        new_img_info['file_name'] = img_info['file_name']
        new_images.append(new_img_info)

        # 保存新标注
        for ann in new_anns:
            new_ann = ann.copy()
            new_ann['id'] = new_ann_id
            new_ann['image_id'] = new_img_info['id']
            new_annotations.append(new_ann)
            new_ann_id += 1

        total_anns_original += len(anns)
        total_anns_new += len(new_anns)
        processed += 1

        if processed % 100 == 0:
            print(f'  已处理 {processed} 张图片...')

    # 保存新标注文件
    output_json = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories'],
    }
    output_json_path = os.path.join(output_ann_dir, f'{args.split}.json')
    with open(output_json_path, 'w') as f:
        json.dump(output_json, f)

    print(f'\n处理完成:')
    print(f'  处理模式: {mode_str}')
    print(f'  数据集划分: {args.split}')
    print(f'  处理图片: {processed}')
    print(f'  跳过图片: {skipped}')
    print(f'  原始标注数: {total_anns_original}')
    print(f'  新标注数: {total_anns_new}')
    print(f'  输出图片目录: {output_img_dir}')
    print(f'  输出标注文件: {output_json_path}')


if __name__ == '__main__':
    main()
