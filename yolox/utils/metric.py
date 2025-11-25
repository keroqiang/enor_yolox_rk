#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import functools
import os
import time
from collections import defaultdict, deque
import psutil

import numpy as np

import torch

__all__ = [
    "AverageMeter",
    "MeterBuffer",
    "get_total_and_free_memory_in_Mb",
    "occupy_mem",
    "gpu_mem_usage",
    "mem_usage"
]


def get_total_and_free_memory_in_Mb(cuda_device):
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        # 无GPU环境下返回默认值
        return 0, 0
    
    try:
        devices_info_str = os.popen(
            "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
        )
        devices_info = devices_info_str.read().strip().split("\n")
        
        # 检查获取的设备信息是否有效
        if not devices_info or devices_info == [''] or len(devices_info) <= int(cuda_device):
            return 0, 0
            
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
            if len(visible_devices) > int(cuda_device):
                cuda_device = int(visible_devices[cuda_device])
        
        # 安全地分割和解析内存信息
        mem_info = devices_info[int(cuda_device)].split(",")
        if len(mem_info) >= 2:
            return int(mem_info[0]), int(mem_info[1])
        return 0, 0
    except Exception:
        # 任何异常情况下都返回默认值
        return 0, 0


def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        return
        
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    # 只有当total大于0时才进行内存预分配
    if total > 0:
        try:
            max_mem = int(total * mem_ratio)
            block_mem = max_mem - used
            # 确保block_mem为正数且合理
            if block_mem > 0 and block_mem < total:
                x = torch.cuda.FloatTensor(256, 1024, min(block_mem, 1024))  # 限制最大分配量
                del x
                time.sleep(5)
        except Exception:
            # 内存分配失败时静默处理
            pass


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def mem_usage():
    """
    Compute the memory usage for the current machine (GB).
    """
    gb = 1 << 30
    mem = psutil.virtual_memory()
    return mem.used / gb


class AverageMeter:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

    @property
    def total(self):
        return self._total

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()


class MeterBuffer(defaultdict):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key="time"):
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()
