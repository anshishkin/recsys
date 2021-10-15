import os
import yaml


def load_yaml(name):
    config_path = os.path.join('configs/', name)
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config