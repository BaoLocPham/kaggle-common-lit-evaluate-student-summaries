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
from metrics import score_loss
from loss import MCRMSELoss
from dataset import read_data, slit_folds, preprocess_text
import lightgbm as lgb

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
    LOGGER.info(configs)
    configs = configs['parameters']
    configs['model']['model_name'] = configs['model']['model_name'].format(select=configs['model']['select'])
    configs['model']['only_model_name'] = configs['model']['only_model_name'].format(select=configs['model']['select'])
    if not configs['debug']:
        wandb.login(key=os.environ['WANDB']) 
        run = wandb.init(entity=config.wandb.entity, project=config.wandb.project, config=configs)
    return wandb

def train_run(model, criterion, optimizer, dataloader):
    gc.collect()
    pass


@torch.no_grad()
def valid_run(model, criterion, dataloader):
    gc.collect()
    pass


def prepare_fold(train, cfg, fold=5):

    dftrain = train[train['fold'] != fold]
    dfvalid = train[train['fold'] == fold]

    pass


def oof_df(fold, true, pred):

    df_pred = pd.DataFrame(pred, columns=['pred_content', 'pred_wording'])
    df_real = pd.DataFrame(true, columns=['content', 'wording'])

    df = pd.concat([df_real, df_pred], axis=1)
    return df


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_main(config):
    config_ = OmegaConf.to_yaml(config)
    # print(config_)
    cfg.__dict__.update(config.parameters)
    prompts_train, prompts_test, summary_train, summary_test, submissions = read_data(
        data_dir=cfg.root_data_dir)
    train = prompts_train.merge(summary_train, on="prompt_id")
    # if cfg.preprocess_text:
    #     LOGGER.info("Performing preprocess text")
    #     train = preprocess_text(train)

    train = slit_folds(train, n_fold=cfg.n_fold, seed=42, strategy=cfg.model.strategy)
    targets = ["content", "wording"]

    drop_columns = ["fold", "student_id", "prompt_id", "text", "fixed_summary_text",
                    "prompt_question", "prompt_title", 
                    "prompt_text"
                ] + targets

    model_dict = {}

    for target in targets:
        models = []
        
        for fold in range(CFG.n_splits):

            X_train_cv = train[train["fold"] != fold].drop(columns=drop_columns)
            y_train_cv = train[train["fold"] != fold][target]

            X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
            y_eval_cv = train[train["fold"] == fold][target]

            dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
            dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

            params = {
                'boosting_type': 'gbdt',
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.048,
                'max_depth': 3,
                'lambda_l1': 0.0,
                'lambda_l2': 0.011
            }

            evaluation_results = {}
            model = lgb.train(params,
                            num_boost_round=10000,
                                #categorical_feature = categorical_features,
                            valid_names=['train', 'valid'],
                            train_set=dtrain,
                            valid_sets=dval,
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=30, verbose=True),
                                lgb.log_evaluation(100),
                                lgb.callback.record_evaluation(evaluation_results)
                                ],
                            )
            models.append(model)
        
        model_dict[target] = models

        # cv
    rmses = []

    for target in targets:
        models = model_dict[target]

        preds = []
        trues = []
        
        for fold, model in enumerate(models):
            X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
            y_eval_cv = train[train["fold"] == fold][target]

            pred = model.predict(X_eval_cv)

            trues.extend(y_eval_cv)
            preds.extend(pred)
            
        rmse = np.sqrt(mean_squared_error(trues, preds))
        print(f"{target}_rmse : {rmse}")
        rmses = rmses + [rmse]

    print(f"mcrmse : {sum(rmses) / len(rmses)}")
if __name__ == "__main__":
    init_experiment()
    train_main()
