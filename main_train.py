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
from dataset import collate, TrainDataset, read_data, slit_folds
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

@hydra.main(version_base=None, config_path="configs", config_name="config")
def init_experiment(config):
    configs = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    # print(configs)
    wandb.login(key=config.wandb.WANDB_API_KEY) 
    LOGGER.info(configs)
    configs = configs['parameters']
    configs['model']['model_name'] = configs['model']['model_name'].format(select=configs['model']['select'])
    configs['model']['only_model_name'] = configs['model']['only_model_name'].format(select=configs['model']['select'])
    run = wandb.init(entity=config.wandb.entity, project=config.wandb.project, config=configs)
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
        batch_size=cfg.model.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.model.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True)

    return train_loader, valid_loader


def oof_df(fold, true, pred):

    df_pred = pd.DataFrame(pred, columns=['pred_content', 'pred_wording'])
    df_real = pd.DataFrame(true, columns=['content', 'wording'])

    df = pd.concat([df_real, df_pred], 1)
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
    # print(config_)
    cfg.__dict__.update(config.parameters)
    prompts_train, prompts_test, summary_train, summary_test, submissions = read_data(
        data_dir=cfg.root_data_dir)
    train = prompts_train.merge(summary_train, on="prompt_id")
    train = slit_folds(train, n_fold=cfg.n_fold, seed=42)
    cfg.model.model_name = cfg.model.model_name.format(select=cfg.model.select)
    cfg.model.only_model_name = cfg.model.only_model_name.format(select=cfg.model.select)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    cfg.tokenizer = tokenizer
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.model.betas = (0.9, 0.999)
    oof_dfs = []
    for fold in range(cfg.n_fold):
        LOGGER.info('\n')
        LOGGER.info(f"========== fold: {fold} training ==========")
        train_loader, valid_loader = prepare_fold(
            train=train, cfg=cfg, fold=fold)
        LOGGER.info(
            f'Number of batches in Train {len(train_loader) } and valid {len(valid_loader)} dataset')
        model = CommontLitModel(cfg.model.model_name, cfg=cfg.model).to(cfg.device)
        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=cfg.model.encoder_lr,
            decoder_lr=cfg.model.decoder_lr,
            weight_decay=cfg.model.weight_decay)
        optimizer = AdamW(
            optimizer_parameters,
            lr=cfg.model.encoder_lr,
            eps=cfg.model.eps,
            betas=cfg.model.betas)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.model.T_max, eta_min=cfg.model.min_lr)

        criterion = nn.SmoothL1Loss(reduction='mean')

        start = time.time()
        best_epoch_score = np.inf
        for epoch in range(cfg.model.num_epoch):

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
                        cfg.save_model_dir,
                        f"{cfg.model.only_model_name}_Fold_{fold}.pth"))

                # saving oof values
                df_ = oof_df(fold, valid_labels, valid_preds)

                LOGGER.info(
                    f'Weights and oof values saved for epochs-{epoch} .....')

            LOGGER.info(
                f"Epoch {epoch} Training Loss {np.round(train_loss , 4)} Validation Loss {np.round(valid_loss , 4)}")
            wandb.log({
                f"Fold {fold} training loss" : np.round(train_loss , 4),
                f"Fold {fold} validation loss" : np.round(valid_loss , 4),
                f"Fold {fold} epoch" : epoch
            })
        end = time.time()
        time_elapsed = end - start

        LOGGER.info(
            ' Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                time_elapsed // 3600, (time_elapsed %
                                       3600) // 60, (time_elapsed %
                                                     3600) %
                60))

        LOGGER.info(" Best Loss: {:.4f}".format(best_epoch_score))

        oof_dfs.append(df_)
        LOGGER.info(
            f" oof for fold {fold} ---> {score_loss(valid_labels, valid_preds )}")
        score_loss_rs = score_loss(valid_labels, valid_preds)
        wandb.log({
                f"Fold {fold} mcrmse_score" : score_loss_rs['mcrmse_score'],
                f"Fold {fold} content_score" : score_loss_rs['content_score'],
                f"Fold {fold} wording_score" : score_loss_rs['wording_score'],
                f"Fold {fold} epoch" : epoch
        })
        del model, train_loader, valid_loader, df_, valid_preds, valid_labels
        gc.collect()
        LOGGER.info('\n')
    oof_df_ = pd.concat(oof_dfs , ignore_index=True )
    average_score = score_loss(np.array(oof_df_[['content' , 'wording']]) , np.array(oof_df_[['pred_content' , 'pred_wording']]))
    LOGGER.info(average_score)
    wandb.log({
        f"Average mcrmse_score" : average_score['mcrmse_score'],
        f"Average content_score" : average_score['content_score'],
        f"Average wording_score" : average_score['wording_score'],
    })
    oof_df_.to_csv(os.path.join(cfg.save_model_dir,
                               'oof_df.csv') , index = False)

if __name__ == "__main__":
    init_experiment()
    train_main()
