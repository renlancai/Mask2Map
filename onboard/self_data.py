
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class SensorCalibration:
    sensor2lidar: Dict[str, float]  # 外参矩阵（translation, rotation）
    sensor2ego: Dict[str, float]
    intrinsic: Dict[str, float]     # 相机内参

@dataclass
class FrameData:
    frame_token: str
    scene_token: str
    timestamp: int
    sensors: Dict[str, dict]        # 各传感器数据
    ego2global: Dict[str, float]    # 全局位姿
    annotations: Dict[str, list]    # 标注信息
    previous_frame_token: Optional[str]

class DatasetProcessor:
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.data_frames = []
        
    # ------------ 待实现接口 ------------
    def _load_rosbag(self, path: Path) -> List[dict]:
        """加载rosbag数据（需用户实现）"""
        pass
    
    def _load_pose(self, path: Path) -> Dict[int, dict]:
        """加载位姿数据（需用户实现）"""
        return {0: {"translation": [0,0,0], "rotation": [0,0,0,1]}}  # 示例数据
    
    def _load_hdmap(self, path: Path) -> dict:
        """加载高精地图（需用户实现）"""
        pass
    
    def _get_nearest_pose(self, poses: Dict[int, dict], timestamp: int) -> dict:
        """获取最近时间戳位姿（需用户实现）"""
        return poses  # 示例实现
    
    def _sync_sensors(self, sensor_data: dict, lidar_timestamp: int) -> dict:
        """时间戳对齐（需用户实现）"""
        return sensor_data  # 示例实现
    # -----------------------------------

    def process_task(self, taskid: str):
        """处理单个taskid数据集"""
        task_path = self.root / taskid
        
        # 加载基础数据
        hdmap = self._load_hdmap(task_path / "hdmap.json")
        gt_pose = self._load_pose(task_path / "poses.bin")
        rosbag_data = self._load_rosbag(task_path / f"{taskid}.bag")
        
        # 初始化帧序列
        previous_token = None
        
        for lidar_frame in filter(lambda x: x['type'] == 'LIDAR', rosbag_data):
            # 构造基础帧
            frame = FrameData(
                frame_token=f"{taskid}_{lidar_frame['timestamp']}",
                scene_token=taskid,
                timestamp=lidar_frame['timestamp'],
                sensors={},
                ego2global=self._get_nearest_pose(gt_pose, lidar_frame['timestamp']),
                annotations={},
                previous_frame_token=previous_token
            )
            
            # 添加LiDAR数据
            frame.sensors['LIDAR'] = {
                'path': str(task_path / f"{lidar_frame['timestamp']}.bin"),
                'extrinsic': load_calibration(taskid).sensor2ego  # 示例
            }
            
            # 同步相机数据
            for cam_data in filter(lambda x: x['type'].startswith('CAM'), rosbag_data):
                synced_data = self._sync_sensors(cam_data, lidar_frame['timestamp'])
                if synced_data:
                    frame.sensors[cam_data['type']] = {
                        'image_path': str(task_path / f"{synced_data['timestamp']}.jpg"),
                        'intrinsic': load_calibration(taskid).intrinsic,
                        'extrinsic': load_calibration(taskid).sensor2lidar
                    }
            
            # 添加标注信息
            frame.annotations.update({
                'obstacles': self._load_obstacles(taskid, frame.timestamp),
                'hdmap_lines': self._extract_hdmap_features(hdmap, frame.ego2global)
            })
            
            # 保存帧数据
            self._save_frame(frame, task_path)
            previous_token = frame.frame_token

    def _save_frame(self, frame: FrameData, task_path: Path):
        """保存帧数据到JSON文件"""
        output_path = task_path / f"{frame.frame_token}.json"
        with open(output_path, 'w') as f:
            json.dump(frame.__dict__, f, indent=2)
        self.data_frames.append(str(output_path))

    def process_all(self):
        """处理所有数据集"""
        for task_dir in self.root.iterdir():
            if task_dir.is_dir():
                self.process_task(task_dir.name)
        
        # 生成总索引文件
        with open(self.root / "all_data_frames.json", 'w') as f:
            json.dump(self.data_frames, f)

if __name__ == "__main__":
    processor = DatasetProcessor(root_path="/data/datasets")
    processor.process_all()


Frame = {
    "CAM_BACK": # N cameras
        instrinsic,
        image_path,
        ego2global_pose(translation, rotation),
        data_token,
        sensor2ego_extrinsic(trans, rot), #fixed for a taskid
        sensor2lidar_extrinsic(trans, rot), #fixed for a taskid
        timestamp,
        type(CAM_BACK),  #CAM_BACK, CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK_RIGHT
        
    timestamp, # cameras may have different timestamps for the triggering event
    frame_token, # token binded with timestamp and maybe taskid
    scene_token, # for a specific taskid?
    lidar_path, # the lidar bin file in binary format, the frame timestamp is the same as the lidar timestamp
    lidar2ego_extrinsic(translation, rotation), #fixed for a taskid
    ego2global_pose(translation, rotation),
    previous_frame_token, # to accumulate the lidar sweeps
}


single collecting dataset:
    taskid, # the taskid for the dataset
    HDMAPMapVersion, # the hdmap version for the dataset
    scene_token = taskid + HDMAPMapVersion, # the task token for the dataset
    hdmap_file, # the hdmap file for the dataset
    rosbag_path, # the rosbag file path, containing the raw data
    extrin_intrin_file, # lidar and cameres extrinsic and intrinsic parameters
    pose_file, # the pose file for the dataset
    obstalces_labeled_file, # the obstacles labeled file for the dataset, one line per frame
    

so processing a multiple taskids datasets, we can:
    
    data_frames = []
    for dataset in datesets:
        rosbag = load_rosbag(f'{root}/{taskid}/{taskid}.bag')
        hdmap = load_hdmap(f'{root}/{taskid}/{hdmap_file}')
        gt_pose = load_gt_pose(f'{root}/{taskid}/{pose_file}')
        all_extrinsic_intrinsics = load_extrinsic_intrinsics(f'{root}/{taskid}/{extrin_intrin_file}')
        obstales = load_obstacles(f'{root}/{taskid}/{obstalces_labeled_file}')
        
        frame_meta = all_extrinsic_intrinsics.fullfill_cameras()
        frame_meta = all_extrinsic_intrinsics.fullfill_lidars()
        
        previous_frame_token = None
        
        for data in rosbag:
            if data.type in ['CAM_BACK', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR']:
                sync_data = sync(data, gt_pose[slice_part])
            if sync_data.is_fullfilled():
                frame['frame_token'] = f'{taskid}_{sync_data.timestamp}'
                frame['scene_token'] = taskid
                frame.synch(sync_data)
                frame.synch(frame_meta)
                frame['ego2global_pose'] = gt_pose.at(frame.timestamp)
                frame['annotation_obstalces'] = obstales(frame.timestamp)
                hdmap_tile = hdmap.get_tile(frame['ego2global_pose'], range=[200, 200])
                frame['annotation_lines'] = construct_lines(hdmap_tile, frame['ego2global_pose'])
                frame['previous_frame_token'] = previous_frame_token
                previous_frame_token = frame['frame_token']
                
                frame_config_path = f'{root}/{frame["frame_token"]}.json'
                json.dump(frame, frame_config_path)
                
                data_frames.append(frame_config_path)

    json.dump(data_frames, f'{root}/all_data_frames.json')