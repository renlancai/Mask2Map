import numpy as np
import cv2
import torch
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d

def bev_seg_to_vectors(bev_seg, class_config, num_points=20):
    vectors = []
    
    # map_classes = ["divider", "ped_crossing", "boundary"] # from Mask2Map
    # map_classes: # from bevfusion
    #     - drivable_area
    #     - ped_crossing
    #     - walkway
    #     - stop_line
    #     - carpark_area
    #     - divider
    
    for cls_name in ['divider']:
        cls_id = class_config[cls_name]
        mask = (bev_seg == cls_id).astype(np.uint8)
        
        # 获取连通域
        contours, _ = cv2.findContours(mask, 
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 转换为标准形状 (N,2)
            points = contour.squeeze(axis=1).astype(np.float32)
            
            # 过滤无效轮廓
            if len(points) < 3:
                continue
                
            # 重采样轮廓
            sampled = sample_contour(points, num_points)
            
            vectors.append({
                'class': cls_name,
                'num_points': num_points,
                'points': sampled
            })
    
    return vectors

def sample_contour(points, target_points):
    """轮廓点重采样函数（修复参数化范围错误）"""
    # 处理闭合轮廓
    if len(points) > 1 and np.allclose(points, points[-1]):
        points = points[:-1]
    
    # 处理无效轮廓
    if len(points) < 3:
        return np.zeros((target_points, 2), dtype=np.float32)
    
    # 计算相邻点间距
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    
    # 构建累积距离（从0开始）
    cum_dists = np.insert(np.cumsum(dists), 0, 0)  # 插入起始0值
    total_length = cum_dists[-1]
    
    # 处理零长度情况（所有点重合）
    if total_length <= 1e-6:
        return np.full((target_points, 2), points, dtype=np.float32)
    
    # 归一化参数化范围到[0,1]
    normalized = cum_dists / total_length
    
    # 创建插值函数（允许外推）
    fx = interp1d(normalized, points[:,0], 
                 kind='linear', 
                 bounds_error=False,
                 fill_value=(points[0,0], points[-1,0]))
    
    fy = interp1d(normalized, points[:,1],
                 kind='linear',
                 bounds_error=False,
                 fill_value=(points[0,1], points[-1,1]))
    
    # 生成采样点
    t = np.linspace(0, 1, target_points)
    return np.column_stack([fx(t), fy(t)]).astype(np.float32)


def compute_bbox(points):
    """
    计算包含所有点的最小轴对齐包围盒
    :param points: 输入点集，形状为(N,2)的数组，N>=1
    :return: (xmin, ymin, xmax, ymax)
    """
    points = np.asarray(points)
    if points.size == 0:
        raise ValueError("输入点集不能为空")
        
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)
    return (xmin.item(), ymin.item(), xmax.item(), ymax.item())


def poseprocess_decode(gt_seg_mask, config, trans_x, trans_y, scale_x, scale_y):
    scores_3d = []
    labels_3d = []
    pts_3d = []
    boxes_3d = []
    total_length = 0
    for k in range(gt_seg_mask.shape[2]):
        temp = gt_seg_mask[:, :, k]
        vectors = bev_seg_to_vectors(temp, config) # divider, ped, boundary
        if (len(vectors) == 0):
            continue
        total_length += len(vectors)
        for vec in vectors:
            scores_3d.extend([1.0])
            labels_3d.extend([k])
            points = vec['points']
            
            points = points - np.array([trans_x, trans_y])
            points = points / np.array([scale_x, scale_y])
            
            pts_3d.append(points)
            boxes_3d.append(compute_bbox(points)) # bbox: xmin, ymin, xmax, ymax
    
    pts_bbox = {}
    pts_bbox['boxes_3d'] = torch.tensor(boxes_3d)
    pts_bbox['scores_3d'] = torch.tensor(scores_3d)
    pts_bbox['labels_3d'] = torch.tensor(labels_3d)
    pts_bbox['pts_3d'] = torch.tensor(pts_3d)

    bbox_list = []
    bbox_list.append({})
    bbox_list[0]["pts_bbox"] = pts_bbox
    return bbox_list

# 示例使用 --------------------------------------------------
if __name__ == "__main__":
    # 创建测试数据
    dummy_seg = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(dummy_seg, (50,100), (200,100), color=1, thickness=5)  # 水平车道线
    cv2.circle(dummy_seg, (150,150), 30, color=2, thickness=2)     # 圆形路沿
    
    # "divider", "ped_crossing", "boundary"
    config = {
        'divider': 1,
        'boundary': 2,
        'ped_crossing': 3,
        'min_area': {
            'divider': 30,
            'boundary': 50,
            'ped_crossing': 50}
    }
    
    # 生成矢量实例
    vectors = bev_seg_to_vectors(dummy_seg, config)
    
    # 验证输出
    for i, vec in enumerate(vectors):
        print(f"Instance {i+1}:")
        print(f"Class: {vec['class']}")
        print(f"Point shape: {vec['points'].shape}")
        print(f"First 3 points:\n{vec['points'][:3]}\n")