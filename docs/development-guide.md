# 开发指南

## 环境准备

### 系统要求

- Python >= 3.6
- PyTorch（需先单独安装，版本根据 CUDA 版本选择）
- CUDA（GPU 训练需要）
- 内存：建议 >= 16GB（开启 RAM 缓存需要更大内存）

### 安装步骤

1. **安装 PyTorch**（先于项目安装）：
```bash
# 参考 https://pytorch.org/ 选择对应版本
pip install torch torchvision
```

2. **安装 YOLOX 项目**：
```bash
cd enor_yolox_rk
pip install -v -e . --no-build-isolation
```

> **重要**：必须使用 `--no-build-isolation`，否则 pip 会在隔离环境中找不到已安装的 torch。

3. **验证安装**：
```bash
python -c "import yolox; print(yolox.__version__)"  # 应输出 0.3.0
```

### 依赖列表

| 包名 | 用途 |
|------|------|
| torch, torchvision | 深度学习框架 |
| numpy | 数值计算 |
| opencv_python | 图像处理 |
| loguru | 日志记录 |
| scikit-image | 图像处理辅助 |
| tqdm | 进度条 |
| Pillow | 图像读写 |
| thop | FLOPs 计算 |
| ninja | C++ 扩展编译加速 |
| tabulate | 表格格式化 |
| tensorboard | 训练可视化 |
| pycocotools | COCO 评估 |
| onnx==1.18 | ONNX 模型导出 |
| onnxruntime==1.18 | ONNX 推理 |
| onnxsim | ONNX 模型简化 |
| mlflow | 实验跟踪（可选） |

## 常用命令

### 训练

```bash
# 基本训练（单GPU）
python tools/train.py -f exps/example/custom/multi_dataset_yolox_s.py \
    -d 1 -b 32 --fp16 -o max_epoch 600

# 从检查点继续训练
python tools/train.py -f exps/example/custom/multi_dataset_yolox_s.py \
    -d 1 -b 32 --fp16 -c YOLOX_outputs/xxx/latest_ckpt.pth --resume

# 使用预训练权重微调
python tools/train.py -f exps/example/custom/multi_dataset_yolox_s.py \
    -d 1 -b 32 --fp16 --ckpt weights/yolox_s.pth -o max_epoch 100
```

### 评估

```bash
python tools/eval.py -d 1 \
    -f exps/example/custom/multi_dataset_yolox_s.py \
    --conf 0.5 \
    -c YOLOX_outputs/multi_dataset_yolox_s/best_ckpt.pth
```

### ONNX 导出

```bash
# 标准导出
python tools/export_onnx.py -f exps/example/custom/xxx.py \
    --output-name model.onnx -n yolox-s -c best_ckpt.pth

# RKNN 优化导出
python tools/export_onnx_new.py -f exps/example/custom/xxx.py \
    --output-name model.onnx -n yolox-s -c best_ckpt.pth --rknpu
```

### 推理演示

```bash
# 图像推理
python tools/demo.py image -n yolox-s -c model.pth \
    --path test.jpg --conf 0.25 --nms 0.45 --tsize 640 \
    --save_result --device gpu

# 视频推理
python tools/demo.py video -n yolox-s -c model.pth \
    --path video.mp4 --conf 0.25 --device gpu
```

## 主要参数说明

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f` | 实验配置文件路径 | - |
| `-d` | GPU 数量 | - |
| `-b` | 批大小 | 64 |
| `--fp16` | 混合精度训练 | False |
| `--cache` | 缓存图片到内存/磁盘 | None |
| `-c` / `--ckpt` | 检查点/预训练权重路径 | None |
| `--resume` | 从检查点恢复训练 | False |
| `-o` | 预占 GPU 内存 | False |
| `-l` | 日志后端 (tensorboard/mlflow/wandb) | tensorboard |
| `opts` | 额外覆盖参数 | - |

### 实验配置覆盖

通过 `opts` 参数可以在命令行覆盖实验配置中的任何属性：

```bash
python tools/train.py -f exps/xxx.py -d 1 -b 32 \
    -o max_epoch 600 num_classes 10 test_conf 0.5
```

## 数据集准备

### COCO 数据集

```bash
cd enor_yolox_rk
mkdir -p datasets/COCO
# 将 COCO 数据集放到此目录：
# datasets/COCO/annotations/instances_train2017.json
# datasets/COCO/annotations/instances_val2017.json
# datasets/COCO/train2017/
# datasets/COCO/val2017/
```

### 自定义数据集

1. 准备 COCO 格式的标注 JSON 文件
2. 在 `exps/example/custom/` 中创建新的实验配置
3. 设置 `self.data_dir`、`self.num_classes`、`self.train_ann`、`self.val_ann`

### 标注处理工具

```bash
# 交互式标注处理（每个数据集目录下都有）
cd datasets/COCO/annotations
python interactive_coco_processor.py
```

功能菜单：
1. 统计数据（类别分布、bbox大小）
2. 类别过滤
3. bbox面积过滤
4. 最大标注数量过滤
5. 质量检查
6. 图像存在性检查
7. 列出所有类别

## 项目结构约定

### 添加新模型变体

1. 在 `exps/default/` 或 `exps/example/custom/` 中创建新的 `.py` 文件
2. 继承 `MyExp`（来自 `yolox_base.py`）
3. 覆盖需要修改的参数（depth、width、num_classes 等）

### 添加新数据集

1. 准备 COCO 格式标注
2. 如果需要类别映射，使用 `MappedCOCODataset`
3. 如果需要多数据集合并，使用 `ConcatDataset`
4. 创建对应的实验配置文件

### 代码风格

- 使用 isort 进行导入排序（行宽100）
- 使用 flake8 进行代码检查（行宽100，复杂度18）
- 遵循 YOLOX 原有代码风格

## 已知问题与解决方案

### 1. pip install 失败

**问题**：`pip install -v -e .` 在隔离构建环境中找不到 torch。

**解决**：使用 `pip install -v -e . --no-build-isolation`

### 2. WSL 下 GPU 检测问题

**问题**：代码使用 `nvidia-smi` 检查 GPU 显存，在 WSL 中可能不工作。

**解决**：已通过代码修改处理 GPU 不可用的情况。

### 3. 类别混淆

**问题**：训练少量类别时，相似类别容易混淆（如 bear → dog）。

**建议**：添加容易混淆的类别参与训练，帮助模型学习区分。

## 开发提示

- 训练输出默认保存在 `YOLOX_outputs/<实验名>/` 目录
- 最佳模型保存为 `best_ckpt.pth`，最新模型保存为 `latest_ckpt.pth`
- 使用 `--fp16` 可显著加速训练并减少显存占用
- 使用 `--cache ram` 将图片缓存到内存加速训练（需要足够内存）
- RKNN 导出时**必须**使用 `--rknpu` 参数，不要修改导出脚本中的默认值
