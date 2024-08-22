import argparse


def get_config():
    parser = argparse.ArgumentParser(description="VisionSCO configuration")
    
    parser.add_argument('--config', default='configs/default.yaml', type=str, help="Path to config yaml file")
    
    args = parser.parse_args()

    return args