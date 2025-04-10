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
    
    # ResetNet50 images only
    # parser.add_argument('--config', help='test config file path',
    #                     default='projects/configs/mask2map/M2M_nusc_r50_full_2Phase_12n12ep.py')
    # parser.add_argument('--checkpoint', help='checkpoint file',
    #                     default='ckpts/r50_phase2.pth')
    
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
    
    print(type(model))
    model = model.cuda().eval()
    raw_model = model.module
    
    logger = get_root_logger()
    # to use the gt
    test_data_root = "/home/tsai/source_code_only/Lidar_AI_Solution/CUDA-BEVFusion/"
    
    def load_mapping(filename):
        """Load token mapping from YAML cache"""
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    #
    token_files = load_mapping(test_data_root + 'data/data_infos/token_mapping.yaml')
    
    raw_results = []
    depart_results = []
    
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        token = data['img_metas'][0].data[0][0]['sample_idx']
        
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
        if (frame_data is None):
            continue
        
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
        
        # print(points.shape)
        # print(lidar2img.shape)
        # print(camera2ego.shape)
        # print(camera_intrinsics.shape)
        # print(img_aug_matrix.shape)
        # print(lidar2ego.shape)
        # print(camera2lidar.shape)
        
        # construct img_metas
        img_metas = {}
        img_metas['img_shape'] = [
            (480, 800, 3), 
            (480, 800, 3),
            (480, 800, 3), 
            (480, 800, 3), 
            (480, 800, 3), 
            (480, 800, 3)]
        
        def tensor2list(input_tensor):
            squeezed_tensor = input_tensor.cpu().squeeze(0) # shape 1*6*4*4 to 6*4*4
            result_list = [squeezed_tensor[i].numpy() for i in range(squeezed_tensor.size(0))]
            return result_list
        
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
        # images_data = data['img'][0].data[0]
        # points = data['points'].data[0]
        
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
            
            # step 1:
            img_feats = raw_model.extract_img_feat(
                img=images_data, img_metas=img_metas, len_queue=None)
            
            # step 2:
            # lidar_feat = raw_model.extract_lidar_feat([points])
            
            # step 3:
            new_prev_bev, bbox_pts = raw_model.simple_test_pts(
                img_feats,
                lidar_feat, # None to disable lidar
                [img_metas],
                prev_bev=None,
                rescale=True)
            bbox_list = [dict() for i in range(len([img_metas]))] # be careful about the len of [img_metas] = 1, not 6
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict["pts_bbox"] = pts_bbox
            depart_results.extend(bbox_list)
            
            ###### way2 :feed raw images to the raw_model, good
            # # NOTIC: images_data shape would be changed if called with extract_img_feat() before
            # _, bbox_list = raw_model.simple_test( # torch.Size([1, 6, 3, 480, 800])
            #     [img_metas], 
            #     images_data,
            #     [points],
            #     prev_bev=None,
            #     rescale=True)
            # depart_results.extend(bbox_list)
            
            prog_bar.update()
            
            visualize_results(data, cfg.point_cloud_range, bbox_list, args, "test_model/")
        if i > 500:
            break
    # you can add the evaluate code here to get mAP data
    do_eval = True
    if do_eval and len(depart_results) > 100:
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
        print(dataset.evaluate(depart_results, **eval_kwargs))
        print("------------thisline---------")

if __name__ == '__main__':
    main()
