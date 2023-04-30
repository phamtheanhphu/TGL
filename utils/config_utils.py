import random
import torch
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")


class Configs:
    def __init__(self, **parsed_configs):
        self.__dict__.update(parsed_configs)


def load_configs(configs_file_path='./config.json'):
    with open(configs_file_path, 'r', encoding='utf-8') as f:
        parsed_configs = json.load(f)
        configs = Configs(**parsed_configs)
        return configs


def init_seed(configs):
    SEED = configs.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
