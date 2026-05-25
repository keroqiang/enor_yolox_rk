# 项目概览

## 项目名称

**enor_yolox_rk** - YOLOX RK NPU 定制版

## 项目简介

本项目是基于 [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) v0.3.0 的定制版本，主要面向 Rockchip NPU 平台的部署优化。YOLOX 是一种 anchor-free 的 YOLO 目标检测算法，采用解耦检测头和 SimOTA 动态标签分配策略。

## 技术栈

| 类别 | 技术 | 版本/说明 |
|------|------|-----------|
| 语言 | Python | >= 3.6 |
| 深度学习框架 | PyTorch | 主要依赖 |
| 计算机视觉 | OpenCV (opencv_python) | 图像处理 |
| 数值计算 | NumPy | 数组操作 |
| 目标检测评估 | pycocotools | COCO mAP 评估 |
| 模型导出 | ONNX (1.18) + onnxruntime (1.18) + onnxsim | ONNX 模型导出和推理 |
| 日志 | loguru | 训练日志 |
| 实验跟踪 | TensorBoard / MLflow / WandB | 可选的实验管理 |
| 模型分析 | thop | FLOPs 计算 |
| 进度条 | tqdm | 训练进度显示 |

## 架构分类

- **仓库类型**: 单体项目 (Monolith)
- **架构模式**: ML 训练框架 + 导出流水线
- **包管理**: pip 安装 (`setup.py`)

## 项目特点（相比原始 YOLOX）

### 1. RKNN NPU 部署适配
- **Focus 模块替换**: 将切片操作替换为标准卷积（`Focus_conv`），兼容 RKNN 工具链
- **SPP 模块优化**: 将大核 MaxPool 拆分为多个 3x3 MaxPool，利于量化
- **后处理分离**: 从模型图中移除后处理，使模型更易于 RKNN 量化和部署
- **支持平台**: rk1808, rv1109, rv1126, rk3399pro, rk3566, rk3562, rk3568, rk3588, rv1103, rv1106

### 2. 多数据集联合训练
- **MappedCOCODataset**: 支持将不同数据集的类别映射到统一 ID 空间
- **ConcatDataset**: 多数据集拼接训练
- **MixedDatasetEvaluator**: 混合数据集评估器

### 3. 类别过滤训练
- 支持 `selected_cat_names` 参数，只训练 COCO 中的指定类别
- 自动过滤不包含目标类别的图像

### 4. MLflow 实验跟踪
- 集成 MLflow 记录训练指标、超参数和模型检查点
- 支持异步日志记录和环境变量配置

### 5. 其他改进
- GPU 内存安全处理（WSL 兼容）
- 使用 `torch.amp` 替代已弃用的 `torch.cuda.amp`
- GPU 显存 95% 预分配限制
- 详细的类别统计打印

## 模型架构

```
输入图像 (640x640x3)
    │
    ▼
┌─────────────────┐
│  CSPDarknet     │  骨干网络（Focus stem + CSP stages）
│  (backbone)     │  输出: dark3(80), dark4(160), dark5(320)
└────────┬────────┘
         │
    ▼         ▼         ▼
┌───────────────────────┐
│  YOLOPAFPN (neck)     │  PAFPN 特征金字塔（FPN + PAN）
│  自顶向下 + 自底向上   │  输出: 3个尺度特征图
└───────────┬───────────┘
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
┌───────────────────────┐
│  YOLOXHead (head)     │  解耦检测头
│  cls分支 + reg分支     │  SimOTA标签分配
│  obj预测              │  IoU Loss + BCE Loss
└───────────────────────┘
    │
    ▼
检测结果 / Loss
```

## 版本信息

- **YOLOX 版本**: 0.3.0
- **基础提交**: `419778480ab6ec0590e5d3831b3afb3b46ab2aa3`
- **许可证**: Apache 2.0
