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
import onnx
from onnxsim import simplify

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument('--rknpu', action="store_true", help='RKNN npu platform')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference

    if os.getenv('RKNN_model_hack', '0') in ['1']:
        from yolox.models.network_blocks import Focus, Focus_conv
        for k,m in model.named_modules():
            if isinstance(m, Focus) and hasattr(m, 'sf'):
                m.sf = Focus_conv()

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    # 设置IR版本为7以兼容旧版本工具
    onnx_version = 7
    
    # 根据PyTorch版本使用不同的导出方式
    if hasattr(torch.onnx, '_export') and sys.version_info >= (3, 8):
        # 针对PyTorch 2.x版本的兼容处理
        torch.onnx._export(
            model,
            dummy_input,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={args.input: {0: 'batch'},
                          args.output: {0: 'batch'}} if args.dynamic else None,
            opset_version=args.opset,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            # 强制使用旧版本导出路径
            use_external_data_format=False,
        )
    else:
        # 标准导出方式
        torch.onnx.export(
            model,
            dummy_input,
            args.output_name,
            input_names=[args.input],
            output_names=[args.output],
            dynamic_axes={args.input: {0: 'batch'},
                          args.output: {0: 'batch'}} if args.dynamic else None,
            opset_version=args.opset,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
        )
    
    # 手动调整IR版本
    onnx_model = onnx.load(args.output_name)
    onnx_model.ir_version = onnx_version
    onnx.save(onnx_model, args.output_name)
    
    # 处理ONNX模型兼容性
    onnx_model = onnx.load(args.output_name)
    
    # 修复Resize节点的输入问题
    for node in onnx_model.graph.node:
        if node.op_type == "Resize":
            # 确保Resize节点有正确的输入数量
            while len(node.input) < 3:
                # 添加空输入占位符
                node.input.append("")
            
            # 移除空的输入
            non_empty_inputs = [inp for inp in node.input if inp]
            node.input[:] = non_empty_inputs
    
    # 保存修复后的模型
    onnx.save(onnx_model, args.output_name)
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        
        # 在简化前确保IR版本兼容
        if onnx_model.ir_version > 7:
            onnx_model.ir_version = 7
            temp_model_path = args.output_name + '.temp'
            onnx.save(onnx_model, temp_model_path)
            onnx_model = onnx.load(temp_model_path)
            os.remove(temp_model_path)
        
        try:
            model_simp, check = simplify(onnx_model,
                                         dynamic_input_shape=args.dynamic,
                                         input_shapes=input_shapes)
            if check:
                onnx.save(model_simp, args.output_name)
                logger.info("generated simplified onnx model named {}".format(args.output_name))
            else:
                logger.warning("Simplified ONNX model could not be validated, using original model")
        except Exception as e:
            logger.warning(f"ONNX simplify failed: {e}, using original model")


if __name__ == "__main__":
    main()
