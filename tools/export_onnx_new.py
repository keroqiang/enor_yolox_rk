#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import sys
from loguru import logger

# activate rknn hack
if len(sys.argv)>=3 and '--rknpu' in sys.argv:
    os.environ['RKNN_model_hack'] = '1'

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

def make_parser():
    parser = argparse.ArgumentParser("YOLOX 充电器数据集 ONNX 导出工具")
    parser.add_argument(
        "--output-name", type=str, default="charger_yolox_s.onnx", help="输出模型名称"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="ONNX模型的输入节点名称"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="ONNX模型的输出节点名称"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="ONNX opset版本"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--dynamic", action="store_true", help="是否使用动态输入形状"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="是否不使用onnxsim简化模型")
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/custom/charger_yolox_s.py",
        type=str,
        help="实验配置文件路径，默认为充电器数据集配置",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="模型名称（保持兼容性）")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="模型检查点路径")
    parser.add_argument('--rknpu', action="store_true", help='是否为RKNN NPU平台优化')
    parser.add_argument(
        "opts",
        help="通过命令行修改配置选项",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="是否在推理时进行解码（推荐启用）"
    )

    return parser

@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("参数配置: {}".format(args))
    
    # 获取充电器数据集的配置
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    
    # 确保正确设置类别数量
    # 从charger_yolox_s.py中我们知道是2个类别
    logger.info(f"当前配置的类别数量: {exp.num_classes}")
    
    # 设置实验名称
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    
    # 创建模型
    model = exp.get_model()
    
    # 设置默认的模型检查点路径
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        if not os.path.exists(ckpt_file):
            # 尝试使用latest_ckpt.pth
            ckpt_file = os.path.join(file_name, "latest_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    
    # 检查检查点文件是否存在
    if not os.path.exists(ckpt_file):
        logger.error(f"未找到模型检查点文件: {ckpt_file}")
        logger.error("请先训练模型或使用正确的检查点路径")
        return
    
    logger.info(f"使用模型检查点: {ckpt_file}")
    
    # 加载模型权重
    ckpt = torch.load(ckpt_file, map_location="cuda", weights_only=False)
    
    model.eval()
    # 处理YOLOX模型保存格式，提取model部分的权重
    if "model" in ckpt:
        ckpt = ckpt["model"]
        logger.info("从checkpoint中提取model权重")
    
    # 加载模型权重到当前模型
    # 对于类别较少的模型，这里确保正确加载权重
    try:
        model.load_state_dict(ckpt)
        logger.info("模型权重加载成功")
    except Exception as e:
        logger.error(f"模型权重加载失败: {e}")
        logger.error("请确认模型检查点与当前配置的类别数量匹配")
        return
    
    # 替换SiLU模块以确保ONNX兼容性
    model = replace_module(model, nn.SiLU, SiLU)
    
    # 设置推理时解码
    model.head.decode_in_inference = args.decode_in_inference
    
    # RKNN平台优化
    if os.getenv('RKNN_model_hack', '0') in ['1']:
        from yolox.models.network_blocks import Focus, Focus_conv
        for k,m in model.named_modules():
            if isinstance(m, Focus) and hasattr(m, 'sf'):
                m.sf = Focus_conv()
        logger.info("已应用RKNN平台优化")
    
    # 创建虚拟输入
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    logger.info(f"创建虚拟输入，形状: {dummy_input.shape}")
    
    # 导出ONNX模型
    logger.info("开始导出ONNX模型...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    logger.success(f"ONNX模型导出成功: {args.output_name}")
    
    # 使用onnxsim简化模型
    if not args.no_onnxsim:
        try:
            import onnx
            from onnxsim import simplify
            
            logger.info("开始使用onnxsim简化模型...")
            input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None
            
            # 加载并简化ONNX模型
            onnx_model = onnx.load(args.output_name)
            model_simp, check = simplify(
                onnx_model,
                dynamic_input_shape=args.dynamic,
                input_shapes=input_shapes
            )
            
            if check:
                onnx.save(model_simp, args.output_name)
                logger.success(f"模型简化成功: {args.output_name}")
            else:
                logger.warning("模型简化验证失败，保留原始模型")
        except ImportError:
            logger.warning("未安装onnxsim，跳过模型简化步骤")
        except Exception as e:
            logger.error(f"模型简化过程出错: {e}")
    
    # 导出完成提示
    logger.info("\n导出完成! 模型信息:")
    logger.info(f"  - 模型路径: {os.path.abspath(args.output_name)}")
    logger.info(f"  - 输入形状: {dummy_input.shape}")
    logger.info(f"  - 类别数量: {exp.num_classes}")
    logger.info(f"  - 推理时解码: {args.decode_in_inference}")
    
    # 使用提示
    logger.info("\n使用提示:")
    logger.info("1. 可以使用Netron工具可视化导出的ONNX模型")
    logger.info("2. 使用ONNX Runtime进行推理测试:")
    logger.info("   python tools/demo.py --onnx charger_yolox_s.onnx --img test.jpg")

if __name__ == "__main__":
    main()