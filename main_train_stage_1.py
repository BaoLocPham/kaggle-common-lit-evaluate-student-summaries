# =========================================================================================
# Libraries
# =========================================================================================
from __future__ import absolute_import
from dataset import (collate, TrainDataset, read_data,
                     read_prompt_grade,
                     preprocess_and_join,
                     slit_folds, preprocess_text, Preprocessor)
import os.path
import warnings
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from tqdm import tqdm
import torch
import gc
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
from transformers import AdamW, AutoTokenizer
import transformers
from metrics import score_loss
from loss import MCRMSELoss
from models import CommontLitModel
from utils import get_logger
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


warnings.filterwarnings("ignore")


LOGGER = get_logger(filename="train_stage_1")
# =========================================================================================
# Configurations
# =========================================================================================


class CFG:
    model = None
    tokenizer = None


cfg = CFG()


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="configs", config_name="config")
def init_experiment(config):
    configs = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    # print(configs)
    LOGGER.info(configs)
    debug = configs['parameters']['debug']
    configs = configs['parameters']['train_stage_1']
    configs['model_name'] = configs['model_name'].format(
        select=configs['select'])
    configs['only_model_name'] = configs['only_model_name'].format(
        select=configs['select'])
    if not debug:
        wandb.login(key=os.environ['WANDB'])
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            group='training_stage_1',
            config=configs)
    return wandb


def train_run(model, criterion, optimizer, dataloader):

    model.train()

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    running_loss = 0.0
    dataset_size = 0.0

    for batch_idx, (data, labels) in bar:
        inputs, target = collate(data), labels
        ids = inputs['input_ids'].to(cfg.device, dtype=torch.long)
        mask = inputs['attention_mask'].to(cfg.device, dtype=torch.long)
        targets = target.to(cfg.device, dtype=torch.float)

        batch_size = ids.size(0)
        outputs = model(ids, mask)
        loss = criterion(outputs, targets)

        # normalize loss to account for batch accumulation
        loss = loss / cfg.model.accum_iter
        loss.backward()

        if ((batch_idx + 1) % cfg.model.accum_iter ==
                0) or (batch_idx + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    gc.collect()
    return epoch_loss


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


def prepare_fold(train, cfg, fold=5):

    dftrain = train[train['fold'] != fold]
    dfvalid = train[train['fold'] == fold]

    train_dataset = TrainDataset(dftrain, cfg=cfg)
    valid_dataset = TrainDataset(dfvalid, cfg=cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_stage_1.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.train_stage_1.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True)

    return train_loader, valid_loader


def oof_df(fold, true, pred):

    df_pred = pd.DataFrame(pred, columns=['pred_content', 'pred_wording'])
    df_real = pd.DataFrame(true, columns=['content', 'wording'])

    df = pd.concat([df_real, df_pred], axis=1)
    return df


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_main(config):
    config_ = OmegaConf.to_yaml(config)
    cfg.__dict__.update(config.parameters)
    prompts_train, prompts_test, summary_train, summary_test, submissions = read_data(
        data_dir=cfg.root_data_dir)
    if cfg.grade_data_dir != "":
        LOGGER.info("Merging with prompt_grade")
        prompt_grade = read_prompt_grade(cfg.grade_data_dir)
        prompts_train = preprocess_and_join(
            prompts_train,
            prompt_grade,
            'prompt_title',
            'title',
            'grade')
    train = prompts_train.merge(summary_train, on="prompt_id")
    # preprocessor = Preprocessor()
    # train = preprocessor.run(prompts_train, summary_train, mode="train")
    # print(train[['prompt_title', 'prompt_question', 'text']])
    if cfg.preprocess_text:
        LOGGER.info("Performing preprocess text")
        train = preprocess_text(train)
    train = slit_folds(train, n_fold=cfg.n_fold, seed=42,
                       strategy=cfg.train_stage_1.strategy)
    cfg.train_stage_1.model_name = cfg.train_stage_1.model_name.format(
        select=cfg.train_stage_1.select)
    cfg.train_stage_1.only_model_name = cfg.train_stage_1.only_model_name.format(
        select=cfg.train_stage_1.select)
    tokenizer = AutoTokenizer.from_pretrained(cfg.train_stage_1.model_name)
    cfg.tokenizer = tokenizer
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.train_stage_1.betas = (0.9, 0.999)
    oof_dfs = []
    for fold in range(cfg.n_fold):
        LOGGER.info('\n')
        LOGGER.info(f"========== fold: {fold} training ==========")
        train_loader, valid_loader = prepare_fold(
            train=train, cfg=cfg, fold=fold)
        LOGGER.info(
            f'Number of batches in Train {len(train_loader) } and valid {len(valid_loader)} dataset')
        model = CommontLitModel(
            cfg.train_stage_1.model_name,
            cfg=cfg.train_stage_1).to(
            cfg.device)
        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=cfg.train_stage_1.encoder_lr,
            decoder_lr=cfg.train_stage_1.decoder_lr,
            weight_decay=cfg.train_stage_1.weight_decay)
        optimizer = AdamW(
            optimizer_parameters,
            lr=cfg.train_stage_1.encoder_lr,
            eps=cfg.train_stage_1.eps,
            betas=cfg.train_stage_1.betas)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train_stage_1.T_max,
            eta_min=cfg.train_stage_1.min_lr)
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=len(train_loader)*0.1*cfg.train_stage_1.num_epoch,
        #     num_training_steps=len(train_loader)*cfg.train_stage_1.num_epoch
        # )

        criterion = MCRMSELoss()

        start = time.time()
        best_epoch_score = np.inf
        for epoch in range(cfg.train_stage_1.num_epoch):

            train_loss = train_run(
                model, criterion, optimizer, dataloader=train_loader)
            valid_loss, valid_preds, valid_labels = valid_run(
                model, criterion, dataloader=valid_loader)

            if valid_loss < best_epoch_score:

                LOGGER.info(
                    f"Validation Loss Improved ({best_epoch_score} ---> {valid_loss})")
                best_epoch_score = valid_loss
                # saving weights
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        cfg.train_stage_1.output_model_dir,
                        f"{cfg.train_stage_1.output_model_name.format(select=cfg.train_stage_1.select, fold=fold)}"))

                # saving oof values
                df_ = oof_df(fold, valid_labels, valid_preds)

                LOGGER.info(
                    f'Weights and oof values saved for epochs-{epoch} .....')

            LOGGER.info(
                f"Epoch {epoch} Training Loss {np.round(train_loss , 4)} Validation Loss {np.round(valid_loss , 4)}")
            if not cfg.debug:
                wandb.log({
                    f"Fold {fold} training loss": np.round(train_loss, 4),
                    f"Fold {fold} validation loss": np.round(valid_loss, 4),
                    f"Fold {fold} epoch": epoch
                })
            score_loss_rs = score_loss(valid_labels, valid_preds)

            if not cfg.debug:
                wandb.log({
                    f"Fold {fold} mcrmse_score": score_loss_rs['mcrmse_score'],
                    f"Fold {fold} content_score": score_loss_rs['content_score'],
                    f"Fold {fold} wording_score": score_loss_rs['wording_score'],
                    f"Fold {fold} epoch": epoch
                })
        end = time.time()
        time_elapsed = end - start

        LOGGER.info(
            ' Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                time_elapsed // 3600, (time_elapsed %
                                       3600) // 60, (time_elapsed %
                                                     3600) %
                60))

        LOGGER.info("Best Loss: {:.4f}".format(best_epoch_score))

        oof_dfs.append(df_)
        LOGGER.info(
            f" oof for fold {fold} ---> {score_loss(valid_labels, valid_preds )}")
        score_loss_rs = score_loss(valid_labels, valid_preds)
        if not cfg.debug:
            wandb.log({
                f"Fold {fold} mcrmse_score": score_loss_rs['mcrmse_score'],
                f"Fold {fold} content_score": score_loss_rs['content_score'],
                f"Fold {fold} wording_score": score_loss_rs['wording_score'],
                f"Fold {fold} epoch": epoch
            })
        del model, train_loader, valid_loader, df_, valid_preds, valid_labels
        gc.collect()
        LOGGER.info('\n')
    oof_df_ = pd.concat(oof_dfs, ignore_index=True)
    average_score = score_loss(np.array(oof_df_[['content', 'wording']]), np.array(
        oof_df_[['pred_content', 'pred_wording']]))
    LOGGER.info(average_score)
    if not cfg.debug:
        wandb.log({
            f"Average mcrmse_score": average_score['mcrmse_score'],
            f"Average content_score": average_score['content_score'],
            f"Average wording_score": average_score['wording_score'],
        })
    oof_df_.to_csv(os.path.join(cfg.save_model_dir,
                                'oof_df.csv'), index=False)


if __name__ == "__main__":
    seed_everything(seed=42)
    init_experiment()
    train_main()
