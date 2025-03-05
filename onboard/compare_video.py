import os.path as osp
import argparse
import os
import glob
import cv2
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('visdir', help='visualize directory')
    parser.add_argument('visdir2', help='visualize directory')

    parser.add_argument('--fps', default=2, type=int, help='fps to generate video')
    parser.add_argument('--video-name', default='compare',type=str)
    parser.add_argument('--sample-name', default='SAMPLE_VIS.jpg', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    parent_dir = osp.join(args.visdir, '..')
    vis_subdir_list = []
    # import pdb;pdb.set_trace()
    size = (1680,450)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path = osp.join(parent_dir,'%s.mp4' % args.video_name)
    video = cv2.VideoWriter(video_path, fourcc, args.fps, size, True)
    file_list = os.listdir(args.visdir)
    
    prog_bar = mmcv.ProgressBar(len(file_list))
    for file in file_list:
        file_path = osp.join(args.visdir, file) 
        if os.path.isdir(file_path):
            surroud_view_path = osp.join(file_path, 'surroud_view.jpg')
            surroud_view_img = cv2.imread(surroud_view_path)
            if (surroud_view_img is None):
                print(f'{surroud_view_path} image is None')
                continue
            
            gt_path = osp.join(file_path, 'GT_fixednum_pts_MAP.png')
            gt_img = cv2.imread(gt_path)
            if (gt_img is None):
                print(f'{gt_path} image is None')
                continue
            
            pred1_path = osp.join(file_path, 'PRED_MAP_plot.png')
            pred1_img = cv2.imread(pred1_path)
            if (pred1_img is None):
                print(f'{pred1_path} image is None')
                continue
            
            file_path2 = osp.join(args.visdir2, file)
            pred2_path = osp.join(file_path2, 'PRED_MAP_plot.png')
            pred2_img = cv2.imread(pred2_path)
            
            if (pred2_img is None):
                print(f'{pred2_path} image is None')
                continue
            
            #
            pred1_img = cv2.copyMakeBorder(pred1_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
            pred2_img = cv2.copyMakeBorder(pred2_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
            gt_img = cv2.copyMakeBorder(gt_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
            
            
            cams_h,cam_w,_ = surroud_view_img.shape
            map_h,map_w,_ = pred1_img.shape
            resize_ratio = cams_h / map_h
            resized_w = map_w * resize_ratio
            resized_pred1_img = cv2.resize(pred1_img,(int(resized_w),int(cams_h)))
            resized_pred2_img = cv2.resize(pred2_img,(int(resized_w),int(cams_h)))
            resized_gt_img = cv2.resize(gt_img,(int(resized_w),int(cams_h)))

            sample_img = cv2.hconcat(
                [surroud_view_img, resized_gt_img, resized_pred1_img, resized_pred2_img])
            
            resized_img = cv2.resize(sample_img, size)
            video.write(resized_img)
        prog_bar.update()
    # import pdb;pdb.set_trace()
    video.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

