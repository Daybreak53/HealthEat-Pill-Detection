import yaml
import os
import glob
from pathlib import Path

# 프로젝트의 최상위 루트 디렉토리 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 설정 파일(config.yaml)의 기본 경로
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config(yaml_path=CONFIG_PATH):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 절대 경로로 변환이 필요한 설정 키 목록
    path_keys = [
        "data_yaml", 
        "train_img_dir", 
        "train_json_dir", 
        "test_img_dir", 
        "output_dir", 
        "submission_csv"
    ]

    # 설정된 상대 경로를 프로젝트 루트 기준의 절대 경로로 업데이트
    for key in path_keys:
        if key in config:
            config[key] = str(PROJECT_ROOT / config[key])

    return config

# 다른 모듈에서 불러와 사용할 전역 설정 객체
CONFIG = load_config()