"""
使用 FiftyOne 下载 Open Images V7 数据集中全量猫(Cat)和狗(Dog)类别的
目标检测数据，并导出为 COCO 格式。

使用前安装依赖：
    pip install fiftyone

用法：
    python tools/download_openimages_cat_dog.py \
        --output_dir datasets/openimages_v7_cat_dog \
        --splits train validation test \
        --export_format coco
"""

import argparse
import os
import fiftyone as fo
import fiftyone.zoo as foz


OPENIMAGES_V7_CLASSES = ["Cat", "Dog"]


def download_and_export(output_dir, splits, export_format):
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        print(f"\n{'='*60}")
        print(f"[1/2] 下载 Open Images V7 - split: {split}")
        print(f"{'='*60}")

        split_export_dir = os.path.join(output_dir, split)

        # 如果 COCO 导出文件已存在，跳过该 split
        coco_json = os.path.join(split_export_dir, "labels.json")
        if os.path.exists(coco_json):
            print(f"  COCO标注已存在: {coco_json}，跳过")
            continue

        dataset_name = f"openimages_v7_cat_dog_{split}"

        # FiftyOne 不支持断点续传，中断后需要清理脏状态重新下载
        if fo.dataset_exists(dataset_name):
            print(f"  检测到未完成的数据集 '{dataset_name}'，清理后重新下载...")
            fo.delete_dataset(dataset_name)

        # 下载全量猫狗类别的目标检测数据
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=OPENIMAGES_V7_CLASSES,
            dataset_name=dataset_name,
        )

        print(f"  下载完成，共 {len(dataset)} 张图片")

        # Open Images V7 的检测标注字段名为 ground_truth
        label_field = "ground_truth"

        # 过滤掉没有标注的样本
        dataset = dataset.exists(label_field)
        print(f"  其中有效标注: {len(dataset)} 张图片")

        if len(dataset) == 0:
            print(f"  跳过空数据集: {split}")
            if isinstance(dataset, fo.DatasetView):
                dataset._dataset.delete()
            else:
                dataset.delete()
            continue

        os.makedirs(split_export_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[2/2] 导出为 COCO 格式 -> {split_export_dir}")
        print(f"{'='*60}")

        dataset.export(
            export_dir=split_export_dir,
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
            classes=OPENIMAGES_V7_CLASSES,
            overwrite=True,
        )

        print(f"  导出完成: {split_export_dir}")

        # 释放当前 split 的数据集以节省内存
        base = dataset._dataset if isinstance(dataset, fo.DatasetView) else dataset
        base.delete()

    print(f"\n{'='*60}")
    print(f"全部完成！数据保存在: {output_dir}")
    print(f"目录结构：")
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            print(f"  {split_dir}/")
            print(f"    ├── data/        (图片)")
            print(f"    └── labels.json  (COCO格式标注)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="下载 Open Images V7 猫狗类别数据（COCO格式）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/openimages_v7_cat_dog",
        help="输出目录 (默认: datasets/openimages_v7_cat_dog)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
        help="要下载的数据集划分 (默认: train validation test)",
    )
    args = parser.parse_args()
    download_and_export(args.output_dir, args.splits, "coco")


if __name__ == "__main__":
    main()
