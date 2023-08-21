# =========================================================================================
# Libraries
# =========================================================================================
from __future__ import absolute_import
import warnings
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from sklearn.metrics import log_loss, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import os.path
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


warnings.filterwarnings("ignore")


# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    pass


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_main(cfg):
    config_ = OmegaConf.to_yaml(cfg)
    print(config_)
    config = CFG()
    config.__dict__.update(cfg.parameters)
