import argparse
import yaml


def get_args() -> dict:
    parser = argparse.ArgumentParser(
        description='EEG Emotion Graph Neural Networks')
    parser.add_argument('--dataset', default='SEED',
                        type=str, help='dataset name', choices=['SEEDIV', 'SEED', 'DREAMER'])
    parser.add_argument('--preset', default='default',
                        type=str, help='Which setting to use for experiment')

    root = '/home/pwjworks/pytorch/emotion_TL-GNNs/config/config.yaml'
    args = parser.parse_args().__dict__
    yaml_data = load_yaml(root)
    config = yaml_data[args['dataset']][args['preset']]
    return config


def load_yaml(path) -> dict:
    yaml_data = None
    print("load config:"+path)
    try:
        with open(path, 'r') as f:
            file_data = f.read()
            yaml_data = yaml.load(file_data, Loader=yaml.Loader)
    except IOError:
        print("load config error")
        exit(1)
    return yaml_data
