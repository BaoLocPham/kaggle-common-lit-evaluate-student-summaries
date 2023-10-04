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


@torch.no_grad()
def valid_run(model, criterion, dataloader):
    model.eval()

    running_loss = 0.0
    dataset_size = 0.0

    predictions = []
    y_labels = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (data, labels) in bar:
        inputs, target = collate(data), labels
        ids = inputs['input_ids'].to(cfg.device, dtype=torch.long)
        mask = inputs['attention_mask'].to(cfg.device, dtype=torch.long)
        targets = target.to(cfg.device, dtype=torch.float)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        predictions.append(outputs.detach().to('cpu').numpy())
        y_labels.append(labels.detach().to('cpu').numpy())

    predictions = np.concatenate(predictions)
    y_labels = np.concatenate(y_labels)
    epoch_loss = running_loss / dataset_size
    gc.collect()

    return epoch_loss, predictions, y_labels


def oof_df(fold, true, pred):

    df_pred = pd.DataFrame(pred, columns=['pred_content', 'pred_wording'])
    df_real = pd.DataFrame(true, columns=['content', 'wording'])

    df = pd.concat([df_real, df_pred], 1)
    return df


@torch.no_grad()
def test_run(model, loader):
    model.eval()
    preds = []
    bar = tqdm(enumerate(loader), total=len(loader))
    for idx, data in bar:
        inputs = collate(data)
        ids = inputs['input_ids'].to(cfg.device, dtype=torch.long)
        mask = inputs['attention_mask'].to(cfg.device, dtype=torch.long)
        y_preds = model(ids, mask)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


@hydra.main(version_base=None, config_path="configs", config_name="config")
def infer_main(config):
    config_ = OmegaConf.to_yaml(config)
    cfg.__dict__.update(config.parameters)
    print(vars(cfg))

    prompts_test, summary_test, submission = read_test(
        data_dir=cfg.root_data_dir)
    if cfg.grade_data_dir != "":
        prompt_grade = read_prompt_grade(cfg.grade_data_dir)
        prompts_test = preprocess_and_join(
            prompts_test,
            prompt_grade,
            'prompt_title',
            'title',
            'grade')
    preprocessor = Preprocessor()
    test = preprocessor.run(prompts_test, summary_test, mode="test")

    if cfg.preprocess_text:
        LOGGER.info("Performing preprocess text")
        test = preprocess_text(test)
    print(test[['prompt_title', 'prompt_question', 'text']])
    LOGGER.info("SAVING preprocessed test dataset")
    test.to_csv(
        os.path.join(
            cfg.prepare_ensemble_inference_stage_1.output_dir,
            cfg.prepare_ensemble_inference_stage_1.output_file
        ),
        index=False
    )


if __name__ == "__main__":
    infer_main()
