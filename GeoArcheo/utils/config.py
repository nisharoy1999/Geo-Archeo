import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Make every path in cfg['paths'] absolute, resolved from GeoArcheo/
    if 'paths' in cfg:
        for key, value in cfg['paths'].items():
            cfg['paths'][key] = os.path.join(PROJECT_ROOT, value)

    return cfg