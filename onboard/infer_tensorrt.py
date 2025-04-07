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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    
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
    
    camera_onnx_file = "onnx/camera_feat_slim.onnx"
    ort_pv_feat = ort.InferenceSession(camera_onnx_file, providers=providers)
    
    bev_downsample_onnx_file = "onnx/camera_bev_downsample.onnx"
    # ort_bev_downsmapler = ort.InferenceSession(bev_downsample_onnx_file, providers=providers)
    
    bev_decoder_onnx_file = "onnx/bev_decoder.onnx"
    # ort_bev_decoder = ort.InferenceSession(bev_decoder_onnx_file, providers=providers)
    
    engine = build_trt(camera_onnx_file, "onnx/camera_feat.trt")
    context = engine.create_execution_context() # affect the cuda-based spconv
    stream = cuda.Stream()
    output_idx = 1
    input_shape = tuple(engine.get_binding_shape(0))
    output_shape = tuple(engine.get_binding_shape(output_idx))
    output_dtype = torch.float16 if engine.get_binding_dtype(output_idx)==trt.DataType.HALF else torch.float32

    # print(input_shape, output_shape, output_dtype)
    
    output_tensor = torch.empty(output_shape).cuda()  



    # while True:
    #     print("start")
    
    raw_results = []
    depart_results = []
    
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
            
            camera_feat_file = f"{tensor_dump_root}/cameras_feat.tensor"
            img_feats = tensor.load(camera_feat_file, return_torch=True).cuda()
            img_feats = img_feats.to(torch.float32)
            
            # validataion: bad
            def torch_to_ort(tensor):
                return ort.OrtValue.from_dlpack(
                    torch.utils.dlpack.to_dlpack(tensor.contiguous()))
            
            # input_name = ort_pv_feat.get_inputs()[0].name
            # output_name = ort_pv_feat.get_outputs()[0].name
            
            # ort_input = torch_to_ort(images_data)
            # io_binding = ort_pv_feat.io_binding()
            
            # io_binding.bind_ortvalue_input(input_name, ort_input)
            # io_binding.bind_output(output_name, 'cuda')  # 输出自动分配在 GPU

            # ort_pv_feat.run_with_iobinding(io_binding)
            # ort_output = io_binding.get_outputs()
            # img_feat_onnx = torch.utils.dlpack.from_dlpack(
            #     ort_output.to_dlpack()
            # )
            
            input_name = ort_pv_feat.get_inputs()[0].name
            ort_output = ort_pv_feat.run(None, {input_name: images_data.cpu().numpy()})
            img_feat_onnx = torch.from_numpy(ort_output[0]).cuda()
            
            img_feat_onnx = img_feat_onnx.to(torch.float32)
            print(f"_________{is_same(img_feat_onnx, img_feats)}")
            max_error = np.max(np.abs((img_feat_onnx - img_feats).detach().cpu().numpy()))
            print(f"max_error: {max_error}")
            
            bindings = [images_data.data_ptr(), output_tensor.data_ptr()]  
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()
            print(is_same(img_feats, output_tensor))
            
            prog_bar.update()
        if i > 10:
            break
    # you can add the evaluate code here to get mAP data
    do_eval = False
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
