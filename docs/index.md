# 项目文档索引

> 生成日期：2026-04-17 | 扫描级别：详尽 (Exhaustive)

## 项目概览

- **项目名称**: enor_yolox_rk
- **类型**: 单体项目 (Monolith) - ML 训练框架库
- **主要语言**: Python
- **架构**: PyTorch 目标检测框架 + RK NPU 部署适配
- **基础项目**: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) v0.3.0

## 快速参考

- **技术栈**: Python 3.6+ / PyTorch / OpenCV / pycocotools / ONNX
- **入口点**: `tools/train.py`（训练）, `tools/eval.py`（评估）, `tools/export_onnx*.py`（导出）
- **架构模式**: 分层实验配置系统 → 模型构建 → 数据加载 → 训练 → 评估 → 导出

## 生成的文档

- [项目概览](./project-overview.md) - 项目简介、技术栈、架构概要
- [架构文档](./architecture.md) - 完整架构说明、数据流、组件关系
- [源码目录结构](./source-tree-analysis.md) - 带注释的完整目录树
- [开发指南](./development-guide.md) - 环境搭建、常用命令、开发约定

## 已有文档

- [主项目说明](../README.md) - YOLOX 原始 README（英文）
- [RKNN 优化说明](../README_rkopt.md) - RKNN 优化简要说明
- [RKNN 优化详细文档](../README_rkopt_manual.md) - RKNN 优化详细说明和平台支持
- [常用命令](../常用命令.md) - 训练/评估/导出命令速查
- [问题记录](../问题记录.md) - 已知问题和解决方案
- [自定义数据训练](./yolox-docs/train_custom_data.md) - 如何使用自定义数据训练
- [训练图片尺寸操作](./yolox-docs/manipulate_training_image_size.md) - 多尺度训练配置
- [模块冻结](./yolox-docs/freeze_module.md) - 冻结模型部分层
- [模型库](./yolox-docs/model_zoo.md) - 标准模型性能基准
- [更新说明](./yolox-docs/updates_note.md) - YOLOX 更新日志
- [快速运行](./yolox-docs/quick_run.md) - 快速上手指南
- [数据集准备](../datasets/README.md) - 数据集目录结构说明

## 快速开始

### 安装

```bash
pip install -v -e . --no-build-isolation
```

### 训练

```bash
python tools/train.py -f exps/example/custom/multi_dataset_yolox_s.py -d 1 -b 32 --fp16 -o max_epoch 600
```

### 评估

```bash
python tools/eval.py -d 1 -f exps/example/custom/multi_dataset_yolox_s.py --conf 0.5 -c YOLOX_outputs/xxx/best_ckpt.pth
```

### RKNN 导出

```bash
python tools/export_onnx_new.py -f exps/example/custom/xxx.py --output-name model.onnx -n yolox-s -c best_ckpt.pth --rknpu
```

## 自定义修改摘要

本项目在原始 YOLOX 基础上的关键修改：

1. **RKNN 部署适配** - Focus 卷积替换、SPP 大核拆分、后处理分离
2. **多数据集训练** - MappedCOCODataset 类别映射 + ConcatDataset
3. **类别过滤** - selected_cat_names 参数支持
4. **MLflow 集成** - 实验跟踪和模型管理
5. **混合精度更新** - torch.amp 替代 torch.cuda.amp
6. **GPU 安全处理** - WSL 兼容的 GPU 检测
