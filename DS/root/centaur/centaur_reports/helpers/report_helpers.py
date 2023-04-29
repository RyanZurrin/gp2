import json

def read_json(config_path):
    with open(config_path, 'rb') as f:
        dict = json.load(f)
    return dict