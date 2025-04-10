# core implementation of the PyTorch inference engine
# divide the whole model into multiple sub-modules(steps):
# include:
#    1. extract camera images features in PV, and ViewTransform to bev
#    2. extract LiDAR point cloud features in BEV
#    3. fusion the BEV features from LIDAR and camera and upsample to get multi-scale BEV feat
#    4. extract the image features in 2D
#    5. generate teh mask-aware queries
#    6. fusion of PQG+GFE to get the final queries with input = [bev-seg-mask, mask-aware-queries]
#    7. mask-guied map decoder to predict the final map elements

import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import warnings
from shapely.errors import ShapelyDeprecationWarning

# from . import tensor
import tensor
import cv2
import numpy as np
import yaml
import shutil

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='module_name')

from pathlib import Path
import os.path as osp

from mmcv_load_images import load_images, mmcv_load_images, visualize_results
from mmcv_load_images import Point, PointFiletr, load_nuscenes_lidar, load_lidar_with_filter_and_transform, PointTransoform

from detection_head import BEVDecoder, CameraFeatures, BEVPooling, CameraFeaturesWithDepth
from detection_head import ImgBEVPooling, FeatFuser, LidarBackboneScn, ImgBEVDownsampling
from detection_head import is_same, compare_two_dicts, global_bev_pooling
from infer_trt import build_trt, get_bindings_info, check_gpu_memory
from voxelize import Voxelization

import onnxruntime as ort
import onnx
from onnx import TensorProto
import ctypes

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from torch.utils.tensorboard import SummaryWriter

def custom_argsort(x, dim):
    values, indices = torch.sort(x, dim=dim)
    return indices

# torch.argsort = custom_argsort

from torch.onnx import register_custom_op_symbolic  

def prim_constant_symbolic(g, value):
    const_tensor = g.op("Constant", value_t=torch.tensor(value))  
    const_tensor.setType(const_tensor.type().with_sizes(value.shape))  
    return const_tensor  


def symbolic_randint(g, low, high, size, dtype=None, **kwargs):
    # 生成[0,1)的随机数
    rand = g.op("RandomUniform", output_i=1, dtype=1, 
                high=1.0, low=0.0, seed=0.0)
    # 缩放并转换为整数
    scaled = g.op("Mul", rand, g.op("Constant", value_t=torch.tensor(high - low)))
    result = g.op("Add", scaled, g.op("Constant", value_t=torch.tensor(low)))
    return g.op("Cast", result, to_i=7)  # to_i=7表示int64


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    
    # v2-99 images only
    # parser.add_argument('--config', help='test config file path',
    #                     default='projects/configs/mask2map/M2M_nusc_v299_full_2Phase_55n55ep.py')
    # parser.add_argument('--checkpoint', help='checkpoint file',
    #                     default='ckpts/v299_110e-df3eb7e5.pth')
    
    # v2-99 images and points
    parser.add_argument('--config', help='test config file path',
                        default='projects/configs/mask2map/M2M_nusc_v299_fusion_full_2Phase_22n22ep_cloud.py')
    parser.add_argument('--checkpoint', help='checkpoint file',
                        default='ckpts/v299_fusion-b0c02deb.pth')
    
    parser.add_argument('--score-thresh', default=0.4, type=float, help='samples to visualize')
    
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='chamfer',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    
    import os
    os.environ["PYTORCH_JIT_LOG_LEVEL"] = "1"  # 启用JIT编译时的详细日志
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # 强制同步CUDA操作以获取准确行号


    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
   
    # ####dataloader & model inference type
    
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
        # nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    model = MMDataParallel(model, device_ids=[0])
    
    print(type(model))  # 可能显示为DataParallel
    model = model.cuda()
    model.eval()

    
    valid_data_token_list = {
        "fd8420396768425eabec9bdddf7e64b6", # for sample0000.yaml
        "30e55a3ec6184d8cb1944b39ba19d622",
        "cc18fde20db74d30825b0b60ec511b7b",
        "08e76760a8c64a92a86686baf68f6aff",
        "2140329a6990437aa46b83c30f49cf49",
        "01a7d01c014f406f87b8fe11976c3d0a",
        "a2fada921a7d4141877f4a51328a21af",
        "b06a815164ec466f9fdb525522bb3799",
        "5bd85334fdf94fb99c7eaa45d5feba0d",
        "61f89208546a4045af336659ebe8db05",
        "3bf56ebb22b741339967a95a9fbe2081",
        "296fcfbf2e29489699f1cb5631f38ff5",
        "f7d75d25c86941f3aecfed9efea1a3e3",
        "1dfecb8189f54b999f4e47ddaa677fd0",
        "830ba619959e4802a955ac40c5ee9453",
        "dc2b67cdadca4deb89d7d684f2894292",
        "e1dffaba060040cfab07dec04790fbfa",
        "be28204a6a5a42ed9939d95ec3f22f5f",
        "77c3d98fab3e4d3ea65747caaa74d605",
        "5224809ffef94a6e83454ad3930d3533",
        "3c18f85f037744b6ae4c8a6f8dc578c2",
        "000681a060c04755a1537cf83b53ba57",
        "000868a72138448191b4092f75ed7776",
        "0017c2623c914571a1ff2a37f034ffd7",
        "00185acff6094c4da3858d78bf462b94",
        "0018da40529f48678419d44d09da5369",
        "0046092508b14f40a86760d11f9896bb",
        "005cfc5e77bc4b60a28499a1fed536c5",
        "005d3e821c4e4e0ab03f1ea1dcbf9cc8",
        "006cbdb235c64999b6f5cfbb63ec88c4",
        "006d520cb38a457fb11e4bf1600a6eb2",
    }
    
    logger = get_root_logger()
    #to use the gt
    test_data_root = "/home/tsai/source_code_only/Lidar_AI_Solution/CUDA-BEVFusion/"
    
    def load_mapping(filename):
        """Load token mapping from YAML cache"""
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    #
    token_files = load_mapping(test_data_root + 'data/data_infos/token_mapping.yaml')
    
    cuda.init()
    device = torch.device('cuda')
    
    providers = ['CUDAExecutionProvider']
    # if 'CUDAExecutionProvider' in ort.get_available_providers():
    #     providers = ['CUDAExecutionProvider']
    # else:
    #     providers = ['CPUExecutionProvider']
    
    camera_onnx_file = "onnx/camera_bev_feat.onnx"
    # ort_bev_feat = ort.InferenceSession(camera_onnx_file, providers=providers)
    
    camera_onnx_file = "onnx/camera_feat.onnx"
    # ort_pv_feat = ort.InferenceSession(camera_onnx_file, providers=providers)
    
    bev_downsample_onnx_file = "onnx/camera_bev_downsample.onnx"
    # ort_bev_downsmapler = ort.InferenceSession(bev_downsample_onnx_file, providers=providers)
    
    bev_decoder_onnx_file = "onnx/bev_decoder.onnx"
    # ort_bev_decoder = ort.InferenceSession(bev_decoder_onnx_file, providers=providers)
    
    # engine = build_trt(camera_onnx_file, "onnx/camera_feat.trt")
    # context = engine.create_execution_context() # affect the cuda-based spconv
    # stream = cuda.Stream()
    # output_idx = 1
    # input_shape = tuple(engine.get_binding_shape(0))
    # output_shape = tuple(engine.get_binding_shape(output_idx))
    # output_dtype = torch.float16 if engine.get_binding_dtype(output_idx)==trt.DataType.HALF else torch.float32

    voxelize_cfg=dict(
        max_num_points=10,
        point_cloud_range=[-15.0, -30.0, -5.0, 15.0, 30.0, 3.0],
        voxel_size=[0.1, 0.1, 0.2],
        max_voxels=[90000, 120000])
    
    voxelizor = Voxelization(
        voxelize_cfg['voxel_size'],
        voxelize_cfg['point_cloud_range'],
        voxelize_cfg['max_num_points'],
        voxelize_cfg['max_voxels']
    )
    
    pfilter = PointFiletr(voxelize_cfg['point_cloud_range'])

    raw_results = []
    depart_results = []
    
    raw_model = model.module
    
    writer = SummaryWriter("logs")
    
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        token = data['img_metas'][0].data[0][0]['sample_idx']
        # if token not in valid_data_token_list:
        #     print(f'skip {token}')
        #     continue
        
        # print(f'item:{i}  processing {token}')
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            print(data['img_metas'][0].data[0][0]['sample_idx'])
            logger.error(f'\n empty gt for index {i}, continue')
            continue
        
        # with torch.no_grad():
        #     result = model(return_loss=False, rescale=True, **data)
        #     raw_results.extend(result)
        #     visualize_results(data, cfg.point_cloud_range, result, args, "good_model/")
        #     prog_bar.update()
        #     continue

        if  token not in token_files:
            continue
        
        filename_stem = token_files[token]
        filename_full = f"{test_data_root}/data/data_infos/samples_info/{filename_stem}.yaml"
        # load only one data
        frame_data = None
        with open(filename_full, 'r') as file:
            frame_data = yaml.load(file, Loader=yaml.FullLoader)
        
        nuscense_data_root = test_data_root
    
        # do some data processing
        scene_token = frame_data["scene_token"]
        frame_token = frame_data["token"]
        if (token != frame_token):
            continue
        
        # points = load_lidar_with_filter_and_transform(nuscense_data_root + frame_data['lidar_path'], pfilter, None)
        # points = torch.tensor(points)
        
        # base_lidar2global = construct(
        #     frame_data['ego2global_rotation'],
        #     frame_data['ego2global_translation'],
        #     frame_data['lidar2ego_rotation'],
        #     frame_data['lidar2ego_translation'])
        
        # for (i in range(max_sweep_num=9)):
        #     previous_token = frame_data['previous_token']
        #     if (previous_token == ""):
        #         break
        #     previous_frame = yaml.load(token_files[previous_token], Loader=yaml.FullLoader)
        #     previous_lidar2global = construct(
        #         previous_frame['ego2global_rotation'],
        #         previous_frame['ego2global_translation'],
        #         previous_frame['lidar2ego_rotation'],
        #         previous_frame['lidar2ego_translation'])
            
        #     transform = base_lidar2global.inverse() * previous_lidar2global
        #     sweep_points = load_lidar_with_filter_and_transform(nuscense_data_root + previous_frame['lidar_path'], pfilter, transform)
        #     points.extend(sweep_points)
            
        
        # load images and points
        tensor_dump_root = test_data_root + "/dump/" + frame_token
        points = tensor.load(f"{tensor_dump_root}/points.tensor", return_torch=True)
        
        # load parameters
        lidar2img = tensor.load(f"{tensor_dump_root}/lidar2image.tensor", return_torch=True)
        camera2ego = tensor.load(f"{tensor_dump_root}/camera2ego.tensor", return_torch=True)
        camera_intrinsics = tensor.load(f"{tensor_dump_root}/camera_intrinsics.tensor", return_torch=True)
        img_aug_matrix = tensor.load(f"{tensor_dump_root}/img_aug_matrix.tensor", return_torch=True)
        lidar2ego = tensor.load(f"{tensor_dump_root}/lidar2ego.tensor", return_torch=True)
        camera2lidar = tensor.load(f"{tensor_dump_root}/camera2lidar.tensor", return_torch=True)
        
        def tensor2list(input_tensor):
            squeezed_tensor = input_tensor.cpu().squeeze(0) # shape 1*6*4*4 to 6*4*4
            result_list = [squeezed_tensor[i].numpy() for i in range(squeezed_tensor.size(0))]
            return result_list
        
        # construct img_metas
        img_metas = {}
        # lidar2img.shape = 1 * 6 * 4 * 4
        img_metas['lidar2img'] = tensor2list(lidar2img)
        img_metas['camera2ego'] = tensor2list(camera2ego)
        img_metas['camera_intrinsics'] = tensor2list(camera_intrinsics)
        img_metas['img_aug_matrix'] = tensor2list(img_aug_matrix)
        img_metas['lidar2ego'] = tensor2list(lidar2ego)
        img_metas['camera2lidar'] = tensor2list(camera2lidar)
        
        temp_img_metas = data['img_metas'][0].data[0][0]
        
        self_data = mmcv_load_images(
            nuscense_data_root,
            frame_data["cams"],
            img_metas['lidar2img'])
        self_image = self_data['img'].unsqueeze(0)
        img_metas['img_aug_matrix'] = self_data['img_aug_matrix']
        
        B, N, w, h, c = self_image.shape
        images_data = self_image.contiguous()
        images_data = images_data.permute(0, 1, 4, 2, 3) # B, N, c, w, h
        # good_images = data['img'][0].data[0]
        # good_lidar_points = data['points'].data[0]
        # points = good_lidar_points
        
        # img_metas = temp_img_metas
        # only overide the nessesary
        # print(img_metas['lidar2img'])
        # print(temp_img_metas['lidar2img'])
        # img_metas['lidar2img'] = temp_img_metas['lidar2img']
        
        # print(img_metas['camera2ego'])
        # print(temp_img_metas['camera2ego'])
        # img_metas['camera2ego'] = temp_img_metas['camera2ego']
        
        # print(img_metas['camera_intrinsics'])
        # print(temp_img_metas['camera_intrinsics'])
        # img_metas['camera_intrinsics'] = temp_img_metas['camera_intrinsics']
        
        # print(img_metas['img_aug_matrix'])
        # print(temp_img_metas['img_aug_matrix'])
        # img_metas['img_aug_matrix'] = temp_img_metas['img_aug_matrix']
        
        # print(img_metas['lidar2ego'])
        # print(temp_img_metas['lidar2ego'])
        # img_metas['lidar2ego'] = temp_img_metas['lidar2ego']
        # img_metas['camera2lidar'] = temp_img_metas['camera2lidar']
        
        with torch.no_grad():
            images_data = images_data.cuda()
            points = points.cuda()
            img_feats = None
            lidar_feat = None

            # #in: B * N * C * W * H, out:  B * N * C * W * H
            # camera_extractor = CameraFeatures(raw_model)
            # camera_extractor.cuda().eval()
            # img_feats = camera_extractor(images_data)
            
            # camera_feat_file = f"{tensor_dump_root}/cameras_feat.tensor"
            # # tensor.save(img_feats, camera_feat_file)
            # img_feats = tensor.load(camera_feat_file, return_torch=True).cuda()
            
            # torch.onnx.export( # bad
            #     camera_extractor, # exported model
            #     images_data, # input tensor
            #     "onnx/camera_feat.onnx",  # onnx path
            #     export_params=True,
            #     opset_version=13,
            #     do_constant_folding=True,
            #     input_names=['img'],
            #     output_names=['img_feats'],
            #     # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            # )
        
            # ort_pv_feat = ort.InferenceSession(camera_onnx_file, providers=providers)
            # input_name = ort_pv_feat.get_inputs()[0].name
            # ort_output = ort_pv_feat.run(None, {input_name: images_data.cpu().numpy()})
            # img_feat_onnx = torch.from_numpy(ort_output[0]).cuda()
            # print(f"_________{is_same(img_feat_onnx, img_feats)}______\n")
            # max_error = np.max(np.abs((img_feat_onnx - img_feats).detach().cpu().numpy()))
            # print(f"max_error: {max_error}")
            
            # writer.add_graph(camera_extractor, images_data)
            # writer.close()
            # break
            
            # img_pooler = ImgBEVPooling(raw_model.pts_bbox_head, img_metas)
            # img_pooler.cuda().eval()
            # bev_embed_img, depth = img_pooler(img_feats)
            
            # img_bev_extractor = CameraFeaturesWithDepth(raw_model, img_metas)
            # img_bev_extractor.cuda().eval()
            # img_feats, enhanced_feats, depth, geom = img_bev_extractor(images_data)
            
            # onnx test: good?
            # input_name = ort_bev_feat.get_inputs()[0].name
            # ort_output = ort_bev_feat.run(None, {input_name: images_data.cpu().numpy()})
            # output_names = [output.name for output in ort_bev_feat.get_outputs()]
            # output_dict = {name: output for name, output in zip(output_names, ort_output)}
            
            # img_feats = torch.from_numpy(output_dict['img_feats']).cuda()
            # enhanced_feats = torch.from_numpy(output_dict['enhanced_feats']).cuda()
            # depth = torch.from_numpy(output_dict['depth']).cuda()
            # geom = torch.from_numpy(output_dict['geom']).cuda()
            
            # print("_____________________")
            # print(is_same(img_feats1, img_feats))
            # print(is_same(enhanced_feats1, enhanced_feats))
            # print(is_same(depth1, depth))
            # print(is_same(geom1, geom))
           
            # torch.onnx.export( # good
            #     img_bev_extractor, # exported model
            #     images_data, # input tensor
            #     "onnx/camera_bev_feat.onnx",  # onnx path
            #     export_params=True,
            #     verbose=False,
            #     opset_version=11,
            #     do_constant_folding=True,
            #     input_names=['img'],
            #     output_names=['img_feats', 'enhanced_feats', 'depth', 'geom'],
            #     dynamic_axes={'img': {0: 'batch_size'},
            #                   'img_feats': {0: 'batch_size'},
            #                   'enhanced_feats': {0: 'batch_size'},
            #                   'depth': {0: 'batch_size'},
            #                   'geom': {0: 'batch_size'}}
            # )
            
            # img_bev_feats = global_bev_pooling(geom, enhanced_feats,
            #             img_bev_extractor.dx,
            #             img_bev_extractor.bx,
            #             img_bev_extractor.nx)
            
            # bev_downsampler = ImgBEVDownsampling(raw_model)
            # bev_downsampler.cuda().eval()
            # bev_embed_img = bev_downsampler(img_bev_feats)
            
            # writer.add_graph(bev_downsampler, img_bev_feats)
            # writer.close()
            # break
            
            # input_name = ort_bev_downsmapler.get_inputs()[0].name
            # ort_output = ort_bev_downsmapler.run(None, {input_name: img_bev_feats.cpu().numpy()})
            # bev_embed_img = torch.from_numpy(ort_output[0]).cuda()
            
            # print("_________" + is_same(bev_embed_img1, bev_embed_img))
            # max_error = np.max(np.abs(bev_embed_img.detach().cpu().numpy() - ort_output[0]))
            # print(f"max_error: {max_error}")
            
            # torch.onnx.export( # bad
            #     bev_downsampler, # exported model
            #     img_bev_feats, # input tensor
            #     "onnx/camera_bev_downsample.onnx",  # onnx path
            #     export_params=True,
            #     opset_version=13,
            #     verbose=True,
            #     do_constant_folding=True,
            #     input_names=['img_bev_feats'],
            #     output_names=['bev_embed_img'],
            #     dynamic_axes={'img_bev_feats': {0: 'batch_size'},
            #                   'bev_embed_img': {0: 'batch_size'}}
            # )
            # break
            

            # trt testing: fail
            # img_feat_trt = torch.empty(output_shape, dtype=output_dtype, device="cuda")
            # bindings = [images_data.data_ptr(), img_feat_trt.data_ptr()]
            # context.execute_async_v2(
            #     bindings=bindings,
            #     stream_handle=stream.handle
            # )
            # stream.synchronize()
            # print(is_same(img_feat_trt, img_feat_onnx))
            
            # # step 2.1.1 good
            
            # f, c, n = voxelizor.forward(points.float())
            # feats = f
            # sizes = None
            # import torch.nn.functional as F
            # coords = F.pad(c, (1, 0), mode="constant", value=0)
            # if n is not None:
            #     sizes = n
 
            # if raw_model.voxelize_reduce and sizes is not None:
            #     feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
            #     feats = feats.contiguous()
            # batch_size = coords[-1, 0] + 1

            # lidar_scn = LidarBackboneScn(raw_model)
            # lidar_scn.cuda().eval()

            # lidar_feat = lidar_scn( \
            #         feats, coords, batch_size, sizes=sizes)
            
            # dummy_input = (feats, coords, batch_size, sizes)
            # writer.add_graph(lidar_scn, dummy_input)
            # writer.close()
            # break
            
            # torch.onnx.export( # test
            #     lidar_scn, # exported model
            #     (feats, coords, batch_size, sizes), # input tensor
            #     "onnx/lidar_scn.onnx",  # onnx path
            #     export_params=True,
            #     opset_version=13,
            #     do_constant_folding=True,
            #     verbose=True,
            #     input_names=['voxel_feat', 'coords', 'bs', 'sizes'],
            #     output_names=['lidar_feat'],
            #     dynamic_axes={'voxel_feat': {0: 'batch_size'},
            #                 'coords': {0: 'batch_size'},
            #                 'sizes': {0: 'batch_size'},
            #                 'lidar_feat': {0: 'batch_size'},
            #                 }
            # )
            # break
            
            # lidar_feat_file = f"{tensor_dump_root}/lidar_feat.tensor"
            # tensor.save(lidar_feat, lidar_feat_file)
            # lidar_feat = tensor.load(lidar_feat_file, return_torch=True).cuda()
            
            # 3.1.1.1 get bev features use lss_bev_encode(future using Cuda code)
            # feat_fuser = FeatFuser(raw_model.pts_bbox_head)
            # feat_fuser.cuda().eval()
            # bev_embed = feat_fuser(bev_embed_img, lidar_feat) #good

            # torch.onnx.export( # good
            #     feat_fuser, # exported model
            #     (bev_embed_img, lidar_feat), # input tensor
            #     "onnx/feat_fuser.onnx",  # onnx path
            #     export_params=True,
            #     opset_version=13,
            #     do_constant_folding=True,
            #     input_names=['img_bev', 'lidar_feat'],
            #     output_names=['fused_bev'],
            #     dynamic_axes={'img_bev': {0: 'batch_size'},
            #                   'lidar_feat': {0: 'batch_size'},
            #                   'img_bev': {0: 'batch_size'}}
            # )

            # onnx_orig = onnx.load("onnx/feat_fuser.onnx")
            # from onnxsim import simplify
            # onnx_simp, check = simplify(onnx_orig)
            # assert check, "Simplified ONNX model could not be validated"
            # onnx.save(onnx_simp, "onnx/feat_fuser.onnx")
            
            bev_feat_file = f"{tensor_dump_root}/bev_feat.tensor"
            # tensor.save(bev_embed, bev_feat_file)
            bev_embed = tensor.load(bev_feat_file, return_torch=True).cuda()
            
            depth_feat_file = f"{tensor_dump_root}/depth.tensor"
            # tensor.save(depth, depth_feat_file)
            # depth = tensor.load(depth_feat_file, return_torch=True).cuda()

            bev_detection_head = BEVDecoder(raw_model.pts_bbox_head, img_metas)
            bev_detection_head.cuda().eval()
            bev_detection_head.forward = bev_detection_head.forward_trt
            classes, coords, pts_coords = bev_detection_head(
                bev_embed)
            
            # input_name0 = ort_bev_decoder.get_inputs()[0].name
            # input_name1 = ort_bev_decoder.get_inputs()[1].name
            # input_name2 = ort_bev_decoder.get_inputs()[2].name
            
            # ort_output = ort_bev_decoder.run(None,
            #     {input_name0: img_feats.cpu().numpy(),
            #         input_name1: bev_embed.cpu().numpy(),
            #         input_name2: depth.cpu().numpy()}
            # )
            
            # classes = torch.from_numpy(ort_output[0]).cuda()
            # coords = torch.from_numpy(ort_output[1]).cuda()
            # pts_coords = torch.from_numpy(ort_output[2]).cuda()
            
            # with torch.no_grad():
            #     torch.onnx.export( # testing
            #         bev_detection_head, # exported model
            #         (img_feats, \
            #             bev_embed, \
            #             depth), # input tensor
            #         "onnx/bev_decoder.onnx",  # onnx path
            #         export_params=True,
            #         # verbose=False,
            #         opset_version=13,
            #         do_constant_folding=True,
            #         input_names=['img_feat', 'bev', 'depth'],
            #         output_names=['classes', 'coords', 'pts_coords'],
            #         dynamic_axes={
            #             'img_feat': {0: 'batch_size'},
            #             'bev': {0: 'batch_size'}, 
            #             'depth': {0: 'batch_size'}, 
            #             'classes': {0: 'batch_size'},
            #             'coords': {0: 'batch_size'},
            #             'pts_coords': {0: 'batch_size'}
            #         },
            #     )
            #     print("test done!\n")
            # break
            
            # post process
            bbox_list_test = bev_detection_head.get_bboxes(classes, coords, pts_coords)
            
            bbox_pts = []
            for bboxes, scores, labels, pts in bbox_list_test: # bbox_list
                result_dict = dict(
                    boxes_3d=bboxes.to("cpu"),
                    scores_3d=scores.cpu(),
                    labels_3d=labels.cpu(),
                    pts_3d=pts.to("cpu"),
                )
                bbox_pts.append(result_dict)
            
            bbox_list = [dict() for i in range(len([img_metas]))] # be careful about the len of [img_metas] = 1, not 6
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict["pts_bbox"] = pts_bbox
            depart_results.extend(bbox_list)
            
            prog_bar.update()
            # visualize_results(data, cfg.point_cloud_range, bbox_list, args, "test_model/")
        if i > 50000:
            break
    # you can add the evaluate code here to get mAP data
    do_eval = True
    if do_eval and len(depart_results) > 5:
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        
        print("------------thisline---------")
        
        # print(dataset.evaluate(raw_results, **eval_kwargs))
        
        print("------------thisline---------")
        
        # eval_kwargs['evaluate_num'] = len(depart_results)
        print(dataset.evaluate(depart_results, **eval_kwargs))
        
        print("------------thisline---------")

if __name__ == '__main__':
    main()
