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
from dataset import collate, TestDataset, read_test
from models import CommontLitModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore")


def get_logger(filename='Training'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()
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
    # print(config_)
    cfg.__dict__.update(config.parameters)
    # cfg.inference.model_name = cfg.model.model_name.format(
    #     select=cfg.model.select)
    # cfg.inference.only_model_name = cfg.model.only_model_name.format(
    #     select=cfg.model.select)
    # cfg.model.model_name = cfg.model.model_name.format(select=cfg.model.select)
    # cfg.model.only_model_name = cfg.model.only_model_name.format(select=cfg.model.select)
    tokenizer = AutoTokenizer.from_pretrained(cfg.inference.model_name)
    cfg.tokenizer = tokenizer
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prompts_test, summary_test, submission = read_test(
        data_dir=cfg.root_data_dir)
    test = prompts_test.merge(summary_test, on="prompt_id")

    test_dataset = TestDataset(test, cfg=cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.inference.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True)
    final_preds = []
    for fold in range(cfg.n_fold):
        print('******** fold', fold, '********')

        model = CommontLitModel(model_name=cfg.inference.model_name, cfg=cfg.inference).to(cfg.device)
        model.load_state_dict(torch.load(
            os.path.join(
                cfg.inference.load_model_dir,
                f"{cfg.inference.only_model_name}_Fold_{fold}.pth"
            ),
            map_location=torch.device('cpu')))
        preds = test_run(model, test_loader)
        final_preds.append(preds)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    final_preds_ = np.mean(final_preds, axis=0)
    target_cols = ['content', 'wording']
    test[target_cols] = final_preds_
    submission = submission.drop(columns=target_cols).merge(
        test[['student_id'] + target_cols], on='student_id', how='left')
    print(submission.head())
    submission[['student_id'] + target_cols].to_csv(
        os.path.join(
            cfg.inference.output_dir, 'submission.csv'
        ), index=False)


if __name__ == "__main__":
    infer_main()
