import torch
from mmcv import Config
from mmdet3d.apis import init_model
import onnx

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings

# 配置文件路径，根据实际情况修改
config_file = 'projects/configs/mask2map/M2M_nusc_r50_full_2Phase_55n55ep.py'
# 模型权重文件路径，根据实际情况修改
checkpoint_file = 'ckpts/55ep_phase2.pth'

# 加载配置
cfg = Config.fromfile(config_file)

# import modules from plguin/xx, registry will be updated
if hasattr(cfg, 'plugin'):
    if cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)
        else:
            # import dir is the dirpath for the config file
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

# 初始化模型
model = init_model(cfg, checkpoint_file, device='cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# 准备示例输入
# BEVFormer 的输入通常包含图像和相机参数等信息
# 这里只是简单示例，实际使用需要根据模型输入要求详细构建
return_loss = False
imgs = [torch.randn(3, 375, 1242).cuda() for _ in range(6)]  # 假设 6 个相机图像
img_metas = [
    {
        'img_shape': [(375, 1242)],
        'ori_shape': [(375, 1242)],
        'cam2img': [torch.randn(4, 4).cuda()],
        'lidar2cam': [torch.randn(4, 4).cuda()],
        'lidar2img': [torch.randn(4, 4).cuda()]
    }
]

inputs = (return_loss, imgs, img_metas)

# 导出模型到 ONNX
output_path = 'ckpts/55ep_phase2.onnx'
torch.onnx.export(
    model,
    inputs,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['images', 'img_metas'],
    output_names=['output'],
    dynamic_axes={
        'images': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# validate the exported onnx file.
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print(f"模型已成功导出到 {output_path}，且 ONNX 模型有效。")
