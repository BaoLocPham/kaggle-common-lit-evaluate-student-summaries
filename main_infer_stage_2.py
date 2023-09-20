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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from metrics import score_loss
from loss import MCRMSELoss
from dataset import read_data, read_test, read_submission, read_data_stage_2, slit_folds, preprocess_text, Preprocessor
import joblib
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
    LOGGER.info(configs)
    configs = configs['parameters']
    configs['model']['model_name'] = configs['model']['model_name'].format(
        select=configs['model']['select'])
    configs['model']['only_model_name'] = configs['model']['only_model_name'].format(
        select=configs['model']['select'])
    if not configs['debug']:
        wandb.login(key=os.environ['WANDB'])
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=configs)
    return wandb


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_main(config):
    config_ = OmegaConf.to_yaml(config)
    cfg.__dict__.update(config.parameters)
    if not cfg.inference_stage_2.have_stage_1:
        prompts_test, summary_test, submission = read_test(
            data_dir=cfg.root_data_dir)
        preprocessor = Preprocessor()
        test = preprocessor.run(prompts_test, summary_test, mode="test")
    else:
        test = pd.read_csv(
            os.path.join(
                cfg.inference_stage_2.input_dir,
                cfg.inference_stage_2.input_file
            )
        )
    drop_columns = [
        "student_id",
        "prompt_id",
        "text",
        "prompt_question",
        "prompt_title",
        "prompt_text",
        "title", "author",
        "description", "genre",
        "path", "date",
        "intro", "excerpt",
        "license", "notes",
        "genre_big_group",
        "grade"
    ]

    model_dict = {}
    targets = ["content", "wording"]

    # Load models
    for target in targets:
        models = []
        for fold in range(cfg.n_fold):
            model = joblib.load(os.path.join(
                cfg.inference_stage_2.input_model_dir,
                cfg.inference_stage_2.input_model_name.format(
                    target=target,
                    fold=fold)))
            models.append(model)
        model_dict[target] = models

    # Inference
    pred_dict = {}
    for target in targets:
        models = model_dict[target]
        preds = []

        for fold, model in enumerate(models):
            X_eval_cv = test.drop(columns=drop_columns)
            # print(X_eval_cv['grade'])


            pred = model.predict(X_eval_cv)
            preds.append(pred)

        pred_dict[target] = preds

    for target in targets:
        preds = pred_dict[target]
        for i, pred in enumerate(preds):
            test[f"{target}_pred_{i}"] = pred
        LOGGER.info(f"target: {target}")
        LOGGER.info(test.head())

        test[target] = test[[f"{target}_pred_{fold}" for fold in range(
            cfg.n_fold)]].mean(axis=1)

    test[["student_id", "content", "wording"]].to_csv(
        os.path.join(
            cfg.inference_stage_2.output_dir, cfg.inference_stage_2.output_file
        ), index=False)

    LOGGER.info("Final Submission")
    LOGGER.info(test[["student_id", "content", "wording"]].head())


if __name__ == "__main__":
    train_main()
