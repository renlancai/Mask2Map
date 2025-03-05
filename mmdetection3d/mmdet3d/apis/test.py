# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp
from mmcv.runner import get_dist_info
import numpy as np

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_gpu_test_onnx(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    #
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    
    # import time
    # time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    
    repetitions = 100
    for i, data in enumerate(data_loader):
        if (i > 10):
            break
        print(data[0])
    
        with torch.no_grad():
            inputs = {}
            inputs['img'] = data['img'][0].data[0].float().unsqueeze(0) #torch.randn(6,3,736,1280)#.cuda()
            #inputs['return_loss'] = False
            inputs['img_metas'] = [1]
            inputs['img_metas'][0] = [1]
            inputs['img_metas'][0][0] = {}
            inputs['img_metas'][0][0]['can_bus'] = torch.from_numpy(data['img_metas'][0].data[0][0]['can_bus']).float()#torch.randn(18)#.cuda()
            inputs['img_metas'][0][0]['lidar2img'] = torch.from_numpy(np.array(data['img_metas'][0].data[0][0]['lidar2img'])).float().unsqueeze(0)#torch.randn(1,6,4,4)#.cuda()
            inputs['img_metas'][0][0]['scene_token'] = 'fcbccedd61424f1b85dcbf8f897f9754'
            inputs['img_metas'][0][0]['img_shape'] = torch.Tensor([[480,800]]) 
            output_file = 'ckpts/v299_110epoch.onnx'
            torch.onnx.export(
                model.module,
                inputs,
                output_file,
                export_params=True,
                keep_initializers_as_inputs=True,
                do_constant_folding=False,
                verbose=True,
                opset_version=11,
            )
 
            print(f"ONNX file has been saved in {output_file}")
            return {0:'1'}
