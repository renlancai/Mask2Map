import os
import yaml
import shutil

def load_mapping(filename):
    """Load token mapping from YAML cache"""
    with open(filename, 'r') as f:
        return yaml.safe_load(f)
    
test_data_root = "/home/tsai/source_code_only/Lidar_AI_Solution/CUDA-BEVFusion/"
token_files = load_mapping(test_data_root + 'data/data_infos/token_mapping.yaml')

# for token in token_files:
#     tensor_dump_root = test_data_root + "/dump/" + token
#     bev_feat_file = f"{tensor_dump_root}/bev_feat.tensor"
#     print(bev_feat_file)
#     os.remove(bev_feat_file)

image_root = "/home/tsai/source_code_only/back_Mask2Map/test_model/"
dest_root = "/home/tsai/source_code_only/back_Mask2Map/all_pictures/"
for token in token_files:
    token_root = image_root + token
    if not os.path.exists(token_root):
        continue
    image_file = f"{token_root}/whole_with_pred_gt.png"
    if not os.path.exists(image_file):
        continue
    
    new_img_path = f"{dest_root}/{token}_pred_gt.png"
    shutil.copyfile(image_file, new_img_path)