import os.path as osp
import argparse
import os
import glob
import cv2
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('visdir', help='visualize directory')
    parser.add_argument('--fps', default=2, type=int, help='fps to generate video')
    parser.add_argument('--video-name', default='demo',type=str)
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
            img_path = osp.join(file_path, 'whole_with_pred_gt.png')
            sample_img = cv2.imread(img_path)
            if (sample_img is None):
                print(f'{img_path} image is None')
                continue
            resized_img = cv2.resize(sample_img, size)
            video.write(resized_img)
        prog_bar.update()
    # import pdb;pdb.set_trace()
    video.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

