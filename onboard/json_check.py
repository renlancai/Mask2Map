import json

# json_data = "/home/tsai/source_code_only/nuscenes/nuscenes_mask2map_anns_val.json"
json_data = "/home/tsai/source_code_only/nuscenes/nuscenes_map_anns_val.json"

data = None
with open(json_data, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
for key in data:
    print(key)

gt_data = data['GTs']

# print(gt_data)
for key in gt_data[0]:
    print(key)

type = type(gt_data[0])

vectors = gt_data[0]['vectors'] # is a list

for vector in vectors:
    print(vector)