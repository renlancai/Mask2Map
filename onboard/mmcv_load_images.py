import argparse
import mmcv
import os
import torch
import warnings
import tensor
import cv2
import numpy as np
import yaml
import shutil

from pathlib import Path
import os.path as osp

def mmcv_load_images(nuscense_data_root, frame_images_data, lidar2img):
    cams_name = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"]

    # step1: LoadMultiViewImageFromFiles
    color_type = 'unchanged'
    to_float32 = True
    
    image_tensors = []
    for cam_name in cams_name:
        path = nuscense_data_root + '/' + frame_images_data[cam_name]['data_path']
        # print(path)
        image = mmcv.imread(path, color_type)
        image_tensors.append(image)
    img = np.stack(image_tensors, axis=-1)
    if to_float32:
        img = img.astype(np.float32)
        
    num_channels = 1 if len(img.shape) < 3 else img.shape[2]

    results = {}
    results['img'] = [img[..., i] for i in range(img.shape[-1])]
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_norm_cfg'] = dict(
        mean=np.zeros(num_channels, dtype=np.float32),
        std=np.ones(num_channels, dtype=np.float32),
        to_rgb=False)
    
    # step 2: RandomScaleImageMultiViewImage
    scales=[0.5]
    rand_ind = np.random.permutation(range(len(scales)))[0]
    rand_scale = scales[rand_ind]
    
    y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
    x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
    
    scale_factor = np.eye(4)
    scale_factor[0, 0] *= rand_scale
    scale_factor[1, 1] *= rand_scale
    results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                        enumerate(results['img'])]
    
    results['lidar2img'] = lidar2img
    lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
    img_aug_matrix = [scale_factor for _ in results['lidar2img']]
    results['lidar2img'] = lidar2img
    results['img_aug_matrix'] = img_aug_matrix
    results['img_shape'] = [img.shape for img in results['img']]
    results['ori_shape'] = [img.shape for img in results['img']]
    # step 3: NormalizeMultiviewImage
    norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True)
    mean = np.array(norm_cfg['mean'])
    std = np.array(norm_cfg['std'])
    
    results['img'] = \
        [mmcv.imnormalize(img, mean, std, norm_cfg['to_rgb']) for img in results['img']]
    results['img_norm_cfg'] = norm_cfg
    
    # padsize to divider(32)
    size_divisor = 32
    pad_val = 0
    
    padded_img = [mmcv.impad_to_multiple(
                img, size_divisor, pad_val=pad_val) for img in results['img']]
    
    results['img'] = padded_img
    results['img_shape'] = [img.shape for img in padded_img]
    results['pad_shape'] = [img.shape for img in padded_img]

    # convert img to Tensor
    combined_np_array = np.stack(results['img'], axis=0)
    results['img'] = torch.from_numpy(combined_np_array)

    # step 4: when test, we donot use MultiScaleFlipAug3D (TTA).
    
    return results

def load_images(nuscense_data_root, frame_images_data):
    cams_name = {
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"}

    image_tensors = []
    for cam_name in cams_name:
        path = nuscense_data_root + '/' + frame_images_data[cam_name]['data_path']
        image = cv2.imread(path)
        image = cv2.resize(image, (800, 480), interpolation=cv2.INTER_AREA)
        if image is None:
            print(f"Failed to read image from {path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)
        image_tensors.append(image_tensor)

    return torch.stack(image_tensors, dim=0)


CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]

def visualize_results(data, pc_range, result, args, out_dir):

    img = data['img'][0].data[0]
    img_metas = data['img_metas'][0].data[0]
    gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
    gt_labels_3d = data['gt_labels_3d'].data[0]

    pts_filename = img_metas[0]['pts_filename']
    pts_filename = osp.basename(pts_filename)
    pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
    
    token = data['img_metas'][0].data[0][0]['sample_idx']
    
    sample_dir = osp.join(out_dir, token)
    mmcv.mkdir_or_exist(osp.abspath(sample_dir))
    
    filename_list = img_metas[0]['filename']
    img_path_dict = {}
    # save cam img for sample
    for filepath in filename_list:
        filename = osp.basename(filepath)
        filename_splits = filename.split('__')
        img_name = filename_splits[1] + '.jpg'
        img_path = osp.join(sample_dir,img_name)
        shutil.copyfile(filepath, img_path)
        img_path_dict[filename_splits[1]] = img_path
    
    # surrounding view
    row_1_list = []
    for cam in CAMS[:3]:
        cam_img_name = cam + '.jpg'
        cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
        lw = 8
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(cam, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        p1 = (0,0)
        p2 = (w,h+3)
        color=(0, 0, 0)
        txt_color=(255, 255, 255)
        cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(cam_img,
                    cam, (p1[0], p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        row_1_list.append(cam_img)
    row_2_list = []
    for cam in CAMS[3:]:
        cam_img_name = cam + '.jpg'
        cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
        if cam == 'BACK':
            cam_img = cv2.flip(cam_img, 1)
        text = cam
        if cam == "CAM_BACK":
            text = f'CAM_BACK {token[:6]}'
        lw = 8
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(text, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        p1 = (0,0)
        p2 = (w,h+3)
        color=(0, 0, 0)
        txt_color=(255, 255, 255)
        cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(cam_img,
                    text, (p1[0], p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        row_2_list.append(cam_img)
    row_1_img=cv2.hconcat(row_1_list)
    row_2_img=cv2.hconcat(row_2_list)
    cams_img = cv2.vconcat([row_1_img,row_2_img])
    
    cams_img_path = osp.join(sample_dir,'surroud_view.jpg')
    cv2.imwrite(cams_img_path, cams_img,[cv2.IMWRITE_JPEG_QUALITY, 70])
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    
    colors_plt = ['orange', 'b', 'r', 'g']
    car_img = Image.open('./figs/lidar_car.png')
    
    from mmdet3d.utils import get_root_logger
    logger = get_root_logger()
    
    gt_fixedpts_map_path = None
    for vis_format in args.gt_format:
        if vis_format == 'se_pts':
            gt_line_points = gt_bboxes_3d[0].start_end_points
            for gt_bbox_3d, gt_label_3d in zip(gt_line_points, gt_labels_3d[0]):
                pts = gt_bbox_3d.reshape(-1,2).numpy()
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
        elif vis_format == 'bbox':
            gt_lines_bbox = gt_bboxes_3d[0].bbox
            for gt_bbox_3d, gt_label_3d in zip(gt_lines_bbox, gt_labels_3d[0]):
                gt_bbox_3d = gt_bbox_3d.numpy()
                xy = (gt_bbox_3d[0],gt_bbox_3d[1])
                width = gt_bbox_3d[2] - gt_bbox_3d[0]
                height = gt_bbox_3d[3] - gt_bbox_3d[1]
                # import pdb;pdb.set_trace()
                plt.gca().add_patch(Rectangle(xy,width,height,linewidth=0.4,edgecolor=colors_plt[gt_label_3d],facecolor='none'))
                # plt.Rectangle(xy, width, height,color=colors_plt[gt_label_3d])
            # continue
        elif vis_format == 'fixed_num_pts':
            plt.figure(figsize=(2, 4))
            plt.xlim(pc_range[0], pc_range[3])
            plt.ylim(pc_range[1], pc_range[4])
            plt.axis('off')
            # gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
            gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
            for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                # import pdb;pdb.set_trace() 
                pts = gt_bbox_3d.numpy()
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                
                plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                plt.scatter(x, y, color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
                # plt.plot(x, y, color=colors_plt[gt_label_3d])
                # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1)
            plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

            gt_fixedpts_map_path = osp.join(sample_dir, 'GT_fixednum_pts_MAP.png')
            plt.savefig(gt_fixedpts_map_path, bbox_inches='tight', format='png',dpi=1200)
            plt.close()   
        elif vis_format == 'polyline_pts':
            plt.figure(figsize=(2, 4))
            plt.xlim(pc_range[0], pc_range[3])
            plt.ylim(pc_range[1], pc_range[4])
            plt.axis('off')
            gt_lines_instance = gt_bboxes_3d[0].instance_list
            # import pdb;pdb.set_trace()
            for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                pts = np.array(list(gt_line_instance.coords))
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                
                # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                # plt.plot(x, y, color=colors_plt[gt_label_3d])
                plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1,alpha=0.8,zorder=-1)
            plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

            gt_polyline_map_path = osp.join(sample_dir, 'GT_polyline_pts_MAP.png')
            plt.savefig(gt_polyline_map_path, bbox_inches='tight', format='png',dpi=1200)
            plt.close()           

        else: 
            logger.error(f'WRONG visformat for GT: {vis_format}')
            raise ValueError(f'WRONG visformat for GT: {vis_format}')
        #
        
        # visualize pred
        # import pdb;pdb.set_trace()
        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']
        pts_3d = result_dic['pts_3d']
        keep = scores_3d > args.score_thresh

        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):

            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)


            pred_bbox_3d = pred_bbox_3d.numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]
            pred_score_3d = float(pred_score_3d)
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()
        
        ##merge into papervis format
        if (gt_fixedpts_map_path == None):
            return
        map_img = cv2.imread(map_path)
        gt_map_img = cv2.imread(gt_fixedpts_map_path)
        map_img = cv2.copyMakeBorder(map_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
        gt_map_img = cv2.copyMakeBorder(gt_map_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)

        cams_h,cam_w,_ = cams_img.shape
        map_h,map_w,_ = map_img.shape
        resize_ratio = cams_h / map_h
        resized_w = map_w * resize_ratio
        resized_map_img = cv2.resize(map_img,(int(resized_w),int(cams_h)))
        resized_gt_map_img = cv2.resize(gt_map_img,(int(resized_w),int(cams_h)))
        
        lw = 8
        tf = max(lw - 1, 1)
        
        w, h = cv2.getTextSize("PRED", 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        p1 = (0,0)
        p2 = (w,h+3)
        color=(0, 0, 0)
        txt_color=(255, 255, 255)
        cv2.rectangle(resized_map_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(resized_map_img,
                    "PRED", (p1[0], p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        
        w, h = cv2.getTextSize("GT", 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        p1 = (0,0)
        p2 = (w,h+3)
        color=(0, 0, 0)
        txt_color=(255, 255, 255)
        cv2.rectangle(resized_gt_map_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(resized_gt_map_img,
                    "GT", (p1[0], p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

        sample_img = cv2.hconcat([cams_img, resized_map_img,resized_gt_map_img])
        
        sample_path = osp.join(sample_dir, 'whole_with_pred_gt.png')
        cv2.imwrite(sample_path, sample_img)
    
    
    # delete temp jpgs to save disk
    for cam in CAMS[:6]:
        cam_img_name = cam + '.jpg'
        img_path = osp.join(sample_dir, cam_img_name)
        os.remove(img_path)
    pass