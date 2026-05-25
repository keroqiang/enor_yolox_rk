# 源码目录结构分析

## 项目根目录

```
enor_yolox_rk/
├── yolox/                    # 核心Python包 - YOLOX框架主体
│   ├── models/               # 模型架构定义
│   │   ├── darknet.py        # CSPDarknet骨干网络（YOLOX主用）
│   │   ├── yolo_pafpn.py     # PAFPN颈部网络（特征金字塔）
│   │   ├── yolo_head.py      # YOLOX解耦检测头 + SimOTA标签分配
│   │   ├── yolox.py          # YOLOX模型组装类（backbone + head）
│   │   ├── yolo_fpn.py       # 传统FPN结构（YOLOv3用）
│   │   ├── network_blocks.py # 基础构建模块（BaseConv/CSPLayer/Focus等）
│   │   ├── losses.py         # IoU/GIoU损失函数
│   │   └── build.py          # 模型构建工厂（按名称创建预训练模型）
│   ├── data/                 # 数据加载与增强
│   │   ├── datasets/
│   │   │   ├── coco.py       # COCO数据集类（扩展：类别过滤）
│   │   │   ├── mapped_coco_dataset.py  # [新增] 带类别映射的COCO数据集
│   │   │   ├── mosaicdetection.py      # Mosaic+MixUp数据增强
│   │   │   ├── datasets_wrapper.py     # 数据集基类/缓存/ConcatDataset
│   │   │   ├── voc.py        # VOC数据集类
│   │   │   ├── coco_classes.py         # COCO 80类名称
│   │   │   └── voc_classes.py          # VOC 20类名称
│   │   ├── data_augment.py   # 数据增强（HSV/仿射/翻转/预处理）
│   │   ├── dataloading.py    # DataLoader定制 + YoloBatchSampler
│   │   ├── data_prefetcher.py # GPU数据预取器
│   │   └── samplers.py       # 无限采样器 + 分布式采样
│   ├── core/                 # 训练核心
│   │   ├── trainer.py        # 训练流程控制器（完整生命周期）
│   │   └── launch.py         # 分布式训练启动器
│   ├── exp/                  # 实验配置系统
│   │   ├── base_exp.py       # 实验配置抽象基类
│   │   ├── yolox_base.py     # YOLOX标准实验配置（默认超参数）
│   │   └── build.py          # 实验配置工厂（动态导入）
│   ├── evaluators/           # 评估器
│   │   ├── coco_evaluator.py       # COCO mAP评估器
│   │   ├── mixed_dataset_evaluator.py  # [新增] 混合数据集评估器
│   │   ├── voc_evaluator.py        # VOC mAP评估器
│   │   └── voc_eval.py             # VOC评估算法实现
│   ├── layers/               # 自定义层
│   │   ├── fast_coco_eval_api.py    # COCO评估C++加速
│   │   └── jit_ops.py              # JIT编译操作框架
│   ├── utils/                # 工具函数
│   │   ├── boxes.py          # 边界框操作 + NMS后处理
│   │   ├── checkpoint.py     # 检查点管理
│   │   ├── ema.py            # 模型指数移动平均
│   │   ├── logger.py         # 日志系统（loguru + WandB）
│   │   ├── lr_scheduler.py   # 学习率调度器（5种策略）
│   │   ├── metric.py         # 指标记录 + GPU内存管理
│   │   ├── model_utils.py    # 模型工具（FLOPs/融合/冻结）
│   │   ├── setup_env.py      # 训练环境配置
│   │   ├── visualize.py      # 检测结果可视化
│   │   ├── allreduce_norm.py # 分布式AllReduce操作
│   │   ├── dist.py           # 分布式通信原语
│   │   ├── compat.py         # PyTorch版本兼容
│   │   ├── demo_utils.py     # 推理演示工具（NMS/后处理）
│   │   └── mlflow_logger.py  # [新增] MLflow实验跟踪
│   └── tools/                # CLI工具包命名空间
├── tools/                    # 命令行工具
│   ├── train.py              # [入口] 训练启动脚本
│   ├── eval.py               # [入口] 评估脚本
│   ├── demo.py               # [入口] 推理演示（图像/视频/摄像头）
│   ├── export_onnx.py        # ONNX导出（支持--rknpu）
│   ├── export_onnx_new.py    # [新增] 增强版ONNX导出（中文界面）
│   ├── export_torchscript.py # TorchScript导出（支持--rknpu）
│   └── trt.py                # TensorRT模型转换
├── exps/                     # 实验配置文件
│   ├── default/              # 标准模型配置
│   │   ├── yolox_s.py        # YOLOX-S (depth=0.33, width=0.50)
│   │   ├── yolox_m.py        # YOLOX-M
│   │   ├── yolox_l.py        # YOLOX-L
│   │   ├── yolox_x.py        # YOLOX-X
│   │   ├── yolox_tiny.py     # YOLOX-Tiny (416x416)
│   │   ├── yolox_nano.py     # YOLOX-Nano (depthwise, 416x416)
│   │   └── yolov3.py         # YOLOv3 (Darknet53 + FPN)
│   └── example/              # 示例配置
│       ├── yolox_voc/        # VOC训练配置
│       └── custom/           # [自定义] 项目专用实验配置
│           ├── multi_dataset_yolox_s.py              # 多数据集联合训练
│           ├── charger_dataset_yolox_s.py            # 充电器单类训练
│           ├── very_small_dataset_yolox_s.py         # 6类训练
│           └── yolox_s_person*_cat_dog_chair*.py     # 不同数据量的7类训练实验
├── datasets/                 # 数据集目录（.gitignore）
│   ├── COCO/                 # COCO数据集
│   │   ├── annotations/      # COCO标注JSON + interactive_coco_processor.py
│   │   ├── train2017/        # 训练图片
│   │   └── val2017/          # 验证图片
│   ├── charger/              # 充电器检测数据集
│   ├── charger_v1/           # 充电器数据集v1 (477张)
│   ├── charger_v2/           # 充电器数据集v2 (1241张, 增强)
│   └── object365_cat_dog/    # Object365猫狗子集
├── demo/                     # 多平台部署演示
│   ├── ONNXRuntime/          # ONNX Runtime推理
│   ├── OpenVINO/             # OpenVINO推理 (Python + C++)
│   ├── MegEngine/            # MegEngine推理 (Python + C++)
│   ├── TensorRT/             # TensorRT推理 (Python + C++)
│   └── ncnn/                 # ncnn推理 (C++ + Android)
├── weights/                  # 预训练权重存放
├── YOLOX_outputs/            # 训练输出（.gitignore）
├── tests/                    # 单元测试
│   └── utils/
│       └── test_model_utils.py  # 模型工具函数测试
├── docs/                     # 项目文档（本次分析生成）
│   ├── index.md              # 文档索引（主入口）
│   ├── project-overview.md   # 项目概览
│   ├── architecture.md       # 架构文档
│   ├── source-tree-analysis.md  # 本文件
│   ├── development-guide.md  # 开发指南
│   ├── project-scan-report.json  # 扫描状态文件
│   └── yolox-docs/           # 原始YOLOX文档
│       ├── train_custom_data.md  # 自定义数据训练教程
│       ├── manipulate_training_image_size.md  # 训练图片尺寸操作
│       ├── freeze_module.md      # 模块冻结教程
│       ├── model_zoo.md          # 模型库
│       ├── updates_note.md       # 更新说明
│       ├── quick_run.md          # 快速运行指南
│       ├── conf.py / Makefile / index.rst  # Sphinx文档构建
│       └── demo/             # 文档中的演示图片
├── setup.py                  # Python包安装配置
├── setup.cfg                 # 代码风格配置 (isort/flake8)
├── requirements.txt          # Python依赖
├── hubconf.py                # PyTorch Hub集成
├── README.md                 # 主项目说明
├── README_rkopt.md           # RKNN优化说明
├── README_rkopt_manual.md    # RKNN优化详细文档
├── 常用命令.md                # 常用命令速查
└── 问题记录.md                # 已知问题记录
```

## 关键入口点

| 入口 | 路径 | 用途 |
|------|------|------|
| 训练 | `tools/train.py` | 启动模型训练 |
| 评估 | `tools/eval.py` | 评估模型精度（mAP） |
| 推理演示 | `tools/demo.py` | 图像/视频/摄像头推理 |
| ONNX导出 | `tools/export_onnx.py` | 导出ONNX模型 |
| ONNX导出(新) | `tools/export_onnx_new.py` | 增强版ONNX导出 |
| TorchScript导出 | `tools/export_torchscript.py` | 导出TorchScript模型 |
| TensorRT转换 | `tools/trt.py` | 转换为TensorRT引擎 |

## 关键目录说明

| 目录 | 职责 | 重要程度 |
|------|------|----------|
| `yolox/models/` | 模型架构（骨干/颈部/检测头/损失） | 核心 |
| `yolox/data/` | 数据加载、增强、数据集定义 | 核心 |
| `yolox/core/` | 训练循环、分布式启动 | 核心 |
| `yolox/exp/` | 实验配置系统 | 核心 |
| `yolox/evaluators/` | 评估指标计算 | 重要 |
| `tools/` | CLI工具和导出脚本 | 重要 |
| `exps/example/custom/` | 项目自定义实验配置 | 项目特有 |
| `demo/` | 多平台部署示例 | 参考 |
