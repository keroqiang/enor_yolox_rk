#!/usr/bin/env python3
"""
分析各数据集在缩放到 640x640 后的目标大小分布
COCO 标准定义：
  小目标: area < 32² = 1024
  中目标: 32² ≤ area < 96² = 9216
  大目标: area ≥ 96²
"""

import json
import os
from collections import defaultdict
from pathlib import Path


SMALL_THRESH = 32 * 32    # 1024
LARGE_THRESH = 96 * 96    # 9216


def analyze_coco_json(json_path, target_size, data_root=None):
    """分析单个 COCO JSON 文件在指定分辨率下的目标大小分布"""
    with open(json_path) as f:
        data = json.load(f)

    images = {img['id']: img for img in data.get('images', [])}
    annotations = data.get('annotations', [])
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    # 按类别统计
    cat_stats = defaultdict(lambda: {'small': 0, 'medium': 0, 'large': 0, 'total': 0})

    for ann in annotations:
        img = images.get(ann.get('image_id'))
        if not img:
            continue

        img_w = img.get('width', 0)
        img_h = img.get('height', 0)
        if img_w <= 0 or img_h <= 0:
            continue

        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            continue

        x, y, w, h = bbox
        # 保持长宽比缩放（和 YOLOX preproc 一致：取 min）
        r = min(target_size / img_w, target_size / img_h)
        scaled_w = w * r
        scaled_h = h * r
        scaled_area = scaled_w * scaled_h

        cat_id = ann.get('category_id', 0)
        cat_name = categories.get(cat_id, f'unknown_{cat_id}')

        if scaled_area < SMALL_THRESH:
            cat_stats[cat_name]['small'] += 1
        elif scaled_area < LARGE_THRESH:
            cat_stats[cat_name]['medium'] += 1
        else:
            cat_stats[cat_name]['large'] += 1
        cat_stats[cat_name]['total'] += 1

    return dict(cat_stats)


def print_stats(name, stats):
    """打印统计结果"""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    if not stats:
        print("  无数据")
        return

    # 汇总
    total_small = sum(s['small'] for s in stats.values())
    total_medium = sum(s['medium'] for s in stats.values())
    total_large = sum(s['large'] for s in stats.values())
    total_all = total_small + total_medium + total_large

    print(f"\n  总计: {total_all:,} 个标注")
    print(f"  ┌──────────┬──────────┬──────────┬──────────┐")
    print(f"  │  类别    │  小目标  │  中目标  │  大目标  │")
    print(f"  │          │ <32²     │ 32²-96²  │ ≥96²     │")
    print(f"  ├──────────┼──────────┼──────────┼──────────┤")

    for cat_name in sorted(stats.keys()):
        s = stats[cat_name]
        t = s['total']
        if t == 0:
            continue
        print(f"  │ {cat_name:8s} │ {s['small']:6,}({s['small']/t*100:4.1f}%) │ {s['medium']:6,}({s['medium']/t*100:4.1f}%) │ {s['large']:6,}({s['large']/t*100:4.1f}%) │")

    print(f"  ├──────────┼──────────┼──────────┼──────────┤")
    print(f"  │ 合计     │ {total_small:6,}({total_small/total_all*100:4.1f}%) │ {total_medium:6,}({total_medium/total_all*100:4.1f}%) │ {total_large:6,}({total_large/total_all*100:4.1f}%) │")
    print(f"  └──────────┴──────────┴──────────┴──────────┘")


def main():
    datasets = [
        # (显示名, json路径)
        ("COCO (完整 train2017)", "datasets/COCO/annotations/instances_train2017.json"),
        ("COCO (person240k+cat+dog)", "datasets/COCO/annotations/coco_train_person240k_cat_dog.json"),
        ("Object365 cat/dog", "datasets/object365_cat_dog/annotations/train_cat_dog.json"),
        ("Open Images V7 cat/dog", "datasets/open_images_v7/annotations/train_cat_dog.json"),
        ("Oxford Pets", "datasets/oxford_pets/annotations/train.json"),
        ("Charger", "datasets/charger/annotations/train_annotations.coco.json"),
    ]

    target_sizes = [480, 640, 800]

    print("各数据集在不同分辨率下的目标大小分布")
    print("COCO 标准: 小目标 <32², 中目标 32²-96², 大目标 ≥96²")
    print(f"训练时多尺度范围: 480 ~ 800 (每10个iteration随机切换)")

    # 只看关注的类别
    focus_cats = {'person', 'cat', 'dog', 'charger', 'Cat', 'Dog'}

    for target_size in target_sizes:
        print(f"\n\n{'#'*70}")
        print(f"  缩放到 {target_size}x{target_size}")
        print(f"{'#'*70}")

        all_stats = {}
        for name, json_path in datasets:
            if not os.path.exists(json_path):
                print(f"\n  [跳过] {name}: 文件不存在")
                continue
            stats = analyze_coco_json(json_path, target_size)
            all_stats[name] = stats
            # 只打印关注的类别
            filtered = {k: v for k, v in stats.items() if k in focus_cats or k.lower() in focus_cats}
            if filtered:
                print_stats(name, filtered)

        # 汇总对比 - 只看 cat/dog
        print(f"\n  --- cat/dog 汇总 ({target_size}x{target_size}) ---")
        print(f"  {'数据集':<35s} │ {'小目标':>8s} │ {'中目标':>8s} │ {'大目标':>8s} │ {'总数':>8s}")
        print(f"  {'─'*35}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")

        for name, stats in all_stats.items():
            cat_small = sum(v['small'] for k, v in stats.items() if k.lower() == 'cat')
            cat_medium = sum(v['medium'] for k, v in stats.items() if k.lower() == 'cat')
            cat_large = sum(v['large'] for k, v in stats.items() if k.lower() == 'cat')
            dog_small = sum(v['small'] for k, v in stats.items() if k.lower() == 'dog')
            dog_medium = sum(v['medium'] for k, v in stats.items() if k.lower() == 'dog')
            dog_large = sum(v['large'] for k, v in stats.items() if k.lower() == 'dog')
            total_s = cat_small + dog_small
            total_m = cat_medium + dog_medium
            total_l = cat_large + dog_large
            total = total_s + total_m + total_l
            if total == 0:
                continue
            short_name = name[:35]
            print(f"  {short_name:<35s} │ {total_s/total*100:6.1f}%  │ {total_m/total*100:6.1f}%  │ {total_l/total*100:6.1f}%  │ {total:>8,}")

    # 最终对比表
    print(f"\n\n{'='*70}")
    print(f"  各数据集 cat/dog 在不同分辨率下的大目标占比")
    print(f"{'='*70}")
    print(f"\n  {'数据集':<35s} │ {'480x480':>10s} │ {'640x640':>10s} │ {'800x800':>10s}")
    print(f"  {'─'*35}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")

    for name, json_path in datasets:
        if not os.path.exists(json_path):
            continue
        row = f"  {name[:35]:<35s} │"
        for target_size in target_sizes:
            stats = analyze_coco_json(json_path, target_size)
            cat_large = sum(v['large'] for k, v in stats.items() if k.lower() == 'cat')
            dog_large = sum(v['large'] for k, v in stats.items() if k.lower() == 'dog')
            cat_total = sum(v['total'] for k, v in stats.items() if k.lower() == 'cat')
            dog_total = sum(v['total'] for k, v in stats.items() if k.lower() == 'dog')
            total = cat_total + dog_total
            large = cat_large + dog_large
            if total > 0:
                row += f" {large/total*100:7.1f}%  │"
            else:
                row += f"    N/A   │"
        print(row)


if __name__ == '__main__':
    main()
