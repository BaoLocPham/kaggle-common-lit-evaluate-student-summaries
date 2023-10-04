# =========================================================================================
# Libraries
# =========================================================================================
from __future__ import absolute_import
import warnings
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import os.path
import sys
import time
from tqdm import tqdm
import torch
import gc
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
from transformers import AdamW, AutoTokenizer
from metrics import score_loss
from dataset import collate, TestDataset, read_prompt_grade, preprocess_and_join, read_data, read_test, preprocess_text, Preprocessor
from models import CommontLitModel
from utils import get_logger
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore")


LOGGER = get_logger(filename="inference_stage_1")
# =========================================================================================
# Configurations
# =========================================================================================


class CFG:
    model = None
    tokenizer = None


cfg = CFG()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def infer_main(config):
    config_ = OmegaConf.to_yaml(config)
    cfg.__dict__.update(config.parameters)

    file_names = cfg.ensemble_inference_stage_1.input_files
    file_weights = cfg.ensemble_inference_stage_1.input_weights
    print(file_names)
    print(file_weights)
    # Define the file paths for your CSV files and their corresponding weights
    # file_paths = ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv']
    # weights = [0.3, 0.2, 0.2, 0.3]  # Adjust these weights as needed

    # Initialize an empty DataFrame to store the ensembled results
    ensembled_df = None

    # Loop through the CSV files and ensemble them
    for i, file_name in enumerate(file_names):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(
            os.path.join(
                cfg.ensemble_inference_stage_1.input_file_dir,
                file_name
            )
        )

            # Apply the weight to the 'content' and 'wording' columns
        df['content'] *= file_weights[i]
        df['wording'] *= file_weights[i]

        # If it's the first iteration, initialize the ensembled DataFrame
        if ensembled_df is None:
            ensembled_df = df
        else:
            # Add the weighted DataFrame to the ensembled DataFrame
            ensembled_df[['content', 'wording']] += df[['content', 'wording']]

    # Divide the ensembled DataFrame by the sum of weights to normalize it
    sum_weights = sum(file_weights)
    ensembled_df[['content', 'wording']] /= sum_weights

    # Save the ensembled DataFrame to a new CSV file
    ensembled_df.to_csv(
        os.path.join(
            cfg.ensemble_inference_stage_1.output_dir,
            cfg.ensemble_inference_stage_1.output_file
        ), index=False)


if __name__ == "__main__":
    infer_main()
