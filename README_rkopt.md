# YOLOX - RKNN optimize
## Source

  Base on https://github.com/Megvii-BaseDetection/YOLOX (v0.3.0) with commit id as 419778480ab6ec0590e5d3831b3afb3b46ab2aa3



## What different

With inference result values unchanged, the following optimizations were applied:

- Optimize focus/SPPF block, getting better performance with same result
- Change output node, remove post_process from the model. (post process block in model is unfriendly for quantization)



## How to use

```
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth --rknpu
```

- Replace 'yolox_s.pth' with your model path
- **NOTICE: Please call with --rknpu param, do not changing the default rknpu value in export.py.** 


## Deploy demo

Please refer https://github.com/airockchip/rknn_model_zoo

