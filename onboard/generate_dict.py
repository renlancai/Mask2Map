import yaml
from pathlib import Path

def generate_token_mapping(directory="."):
    """Generate {token: filename} mapping from YAML files"""
    token_dict = {}
    i = 0
    for filepath in Path(directory).glob("*.yaml"):
        filename = filepath.stem  # Get filename without extension
        
        with open(filepath, 'r') as f:
            # data = yaml.safe_load(f)
            data = yaml.load(f, Loader=yaml.FullLoader)
            if not data or 'token' not in data:
                raise ValueError(f"Missing 'token' field in file: {filepath.name}")
            
            token = data['token']
            if token in token_dict:
                existing_file = token_dict[token]
                raise ValueError(f"Token conflict: '{token}' exists in both '{filename}' and '{existing_file}'")
            i += 1
            print(f'{i} -- ' + token + " ---  " + filename)
            token_dict[token] = filename
    
    return token_dict

if __name__ == "__main__":
    try:
        test_data_root = "/home/tsai/source_code_only/Lidar_AI_Solution/CUDA-BEVFusion/"
        
        mapping = generate_token_mapping(\
            test_data_root + 'data/data_infos/samples_info/')
        # dump to file
        with open(test_data_root + 'data/data_infos/token_mapping.yaml', 'w') as f:
            yaml.dump(mapping, f)
    except ValueError as e:
        print(f"🚨 Error: {e}")
