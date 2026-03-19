import yaml
import os
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config(yaml_path=CONFIG_PATH):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    path_keys = [
        "data_yaml", 
        "train_img_dir", 
        "train_json_dir", 
        "test_img_dir", 
        "output_dir", 
        "submission_csv"
    ]

    for key in path_keys:
        if key in config:
            config[key] = str(PROJECT_ROOT / config[key])

    return config

CONFIG = load_config()