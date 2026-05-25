# 架构文档

## 1. 执行摘要

enor_yolox_rk 是一个基于 YOLOX v0.3.0 的目标检测框架定制版本，核心功能包括模型训练、评估、ONNX/TorchScript 导出以及 RK NPU 平台部署适配。项目采用 Python 包架构，通过实验配置系统（`exp/`）管理不同模型变体和训练参数。

## 2. 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.6+ | 项目主体语言 |
| 深度学习 | PyTorch | 模型定义、训练、推理 |
| 计算机视觉 | OpenCV | 图像处理与可视化 |
| 评估 | pycocotools | COCO mAP 评估指标 |
| 导出 | ONNX 1.18 | 模型导出格式 |
| 简化 | onnxsim | ONNX 模型简化 |
| 日志 | loguru | 训练过程日志 |
| 实验跟踪 | TensorBoard / MLflow / WandB | 实验管理（可选） |
| 测试 | unittest | 单元测试（最小覆盖） |

## 3. 架构模式

项目采用**分层实验配置模式**，核心流程为：

```
实验配置 (exp) → 模型构建 (models) → 数据加载 (data) → 训练循环 (core) → 评估 (evaluators) → 导出 (tools)
```

### 3.1 实验配置系统

```
BaseExp (抽象基类，定义接口)
  └── Exp / yolox_base.py (YOLOX默认配置)
       ├── exps/default/yolox_s.py (标准模型变体)
       └── exps/example/custom/*.py (项目自定义配置)
```

每个实验配置类定义了完整的训练参数：模型结构、数据路径、增强策略、学习率调度、评估方式等。通过 `-f` 参数指定配置文件。

### 3.2 模型架构

```
YOLOX
├── backbone: CSPDarknet
│   ├── Focus stem (空间→通道转换)
│   ├── Stage 1-3: CSPLayer + 下采样
│   └── Stage 4: CSPLayer + SPPBottleneck
├── neck: YOLOPAFPN
│   ├── FPN (自顶向下): dark5 → dark4 → dark3
│   └── PAN (自底向上): dark3 → dark4 → dark5
└── head: YOLOXHead (三个尺度共享)
    ├── cls_convs → cls_preds (分类)
    ├── reg_convs → reg_preds (回归)
    ├── obj_preds (目标性)
    └── SimOTA标签分配 (训练时)
```

### 3.3 数据流水线

```
COCODataset / MappedCOCODataset / VOCDetection
    │
    ▼ (可选缓存: RAM / 磁盘)
MosaicDetection (Mosaic 4图拼接 + MixUp增强)
    │
    ▼ (HSV增强 + 仿射变换 + 翻转)
TrainTransform / ValTransform
    │
    ▼ (预处理: 缩放 + padding + 转置)
DataLoader (YoloBatchSampler + InfiniteSampler)
    │
    ▼ (DataPrefetcher GPU预取)
训练循环
```

### 3.4 训练流程

```
tools/train.py
    │
    ▼ launch() - 分布式启动
Trainer.train()
    ├── before_train(): 初始化模型、优化器、数据加载器、EMA、日志
    ├── train_in_epoch() × max_epoch:
    │   ├── before_epoch(): 后期关闭Mosaic + 启用L1 Loss
    │   ├── train_one_iter(): 前向 → 反向 → 更新 → EMA
    │   └── after_epoch(): 保存检查点 + 周期评估
    └── after_train(): 最终评估 + 最佳模型保存
```

### 3.5 RKNN 导出流水线

```
训练模型 (.pth)
    │
    ▼ export_onnx.py --rknpu
    ├── 设置 RKNN_model_hack=1 环境变量
    ├── Focus → Focus_conv (卷积替换切片)
    ├── SPPBottleneck 大核拆分
    ├── YOLOXHead 跳过后处理
    ├── ONNX导出 + IR版本修正
    └── onnxsim 简化
    │
    ▼
ONNX模型 (.onnx)
    │
    ▼ RKNN Toolkit (外部工具)
RKNN模型 (.rknn) → 部署到 Rockchip NPU
```

## 4. 数据架构

### 4.1 数据集格式

项目使用 COCO JSON 标注格式，标准目录结构：

```
datasets/
├── COCO/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── train2017/
│   └── val2017/
├── charger/
│   ├── annotations/
│   │   └── instances_default.json
│   └── images/
└── object365_cat_dog/
    ├── annotations/
    └── images/
```

### 4.2 类别映射系统

多数据集训练时，通过 `MappedCOCODataset` 的 `class_mapping` 参数将不同数据集的原始类别名映射到统一 ID：

```python
class_mapping = {
    "person": 0, "cat": 1, "dog": 2,
    "chair": 3, "couch": 4, "bird": 5,
    "charger": 6
}
```

### 4.3 标注处理工具

`interactive_coco_processor.py` 提供交互式标注处理功能：
- 统计分析（类别分布、bbox 大小分布）
- 类别过滤
- bbox 面积过滤
- 质量检查（越界/过小/重复/无效）
- 自动修复
- 图像存在性验证

## 5. 自定义组件清单

| 组件 | 文件 | 说明 |
|------|------|------|
| MappedCOCODataset | `yolox/data/datasets/mapped_coco_dataset.py` | 类别映射数据集 |
| MixedDatasetEvaluator | `yolox/evaluators/mixed_dataset_evaluator.py` | 混合数据集评估器 |
| MlflowLogger | `yolox/utils/mlflow_logger.py` | MLflow 实验跟踪 |
| Focus_conv | `yolox/models/network_blocks.py` | RKNN兼容Focus模块 |
| interactive_coco_processor | `datasets/*/annotations/` | 标注处理工具 |
| export_onnx_new.py | `tools/export_onnx_new.py` | 增强版ONNX导出 |
| 自定义实验配置 | `exps/example/custom/*.py` | 多种训练实验配置 |

## 6. 开发工作流

### 6.1 安装

```bash
cd enor_yolox_rk
pip install -v -e . --no-build-isolation
```

> 注意：`--no-build-isolation` 是必需的，因为 pip 隔离构建环境无法找到已安装的 torch。

### 6.2 训练

```bash
python tools/train.py -f exps/example/custom/multi_dataset_yolox_s.py \
    -d 1 -b 32 --ckpt weights/yolox_s.pth --fp16 -o max_epoch 600
```

### 6.3 评估

```bash
python tools/eval.py -d 1 \
    -f exps/example/custom/multi_dataset_yolox_s.py \
    --conf 0.5 -c YOLOX_outputs/multi_dataset_yolox_s/best_ckpt.pth
```

### 6.4 ONNX导出

```bash
python tools/export_onnx_new.py \
    -f exps/example/custom/multi_dataset_yolox_s.py \
    --output-name model.onnx -n yolox-s \
    -c path/to/best_ckpt.pth --rknpu
```

## 7. 测试策略

当前测试覆盖非常有限：

| 测试文件 | 覆盖范围 |
|----------|----------|
| `tests/utils/test_model_utils.py` | `adjust_status` 和 `freeze_module` 函数 |

**未覆盖区域**：数据集加载、模型构建、训练循环、ONNX 导出、RKNN 转换、评估器、端到端推理。

## 8. 部署架构

```
训练 (PyTorch GPU)
    │
    ▼ .pth 检查点
ONNX 导出 (tools/export_onnx*.py --rknpu)
    │
    ▼ .onnx 模型
┌─────────────────────────┐
│ 外部部署流水线           │
│ RKNN Toolkit → .rknn    │ → Rockchip NPU 设备
│ TensorRT → .engine      │ → NVIDIA GPU
│ ONNX Runtime → 直接推理  │ → CPU/GPU
│ OpenVINO → .xml/.bin    │ → Intel 设备
│ ncnn → .param/.bin      │ → 移动端
└─────────────────────────┘
```
