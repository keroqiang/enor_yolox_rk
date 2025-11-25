# YOLOX - RKNN optimize
## Source

  Base on https://github.com/Megvii-BaseDetection/YOLOX (v0.3.0) with commit id as 419778480ab6ec0590e5d3831b3afb3b46ab2aa3



## What different

Inference result unchanged:

- Optimize focus/SPPF block, getting better performance with same result
- Change output node, remove post_process from the model. (post process is unfriendly in quantization)



## How to use

```
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth --rknpu {rk_platform}
or
python3 tools/export_torchscript.py --output-name yolox_s.pt -n yolox-s -c yolox_s.pth --rknpu {rk_platform}
```

- rk_platform support  rk1808, rv1109, rv1126, rk3399pro, rk3566, rk3562, rk3568, rk3588, rv1103, rv1106. (Actually the exported models are the same in spite of the exact platform )

- Replace 'yolox_s.pth' with your model path
- **NOTICE: Please call with --rknpu param, do not changing the default rknpu value in export.py.** 

