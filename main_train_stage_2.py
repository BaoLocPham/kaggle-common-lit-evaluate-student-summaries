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
from dataset import read_data, read_prompt_grade, preprocess_and_join, read_data_stage_2, slit_folds, preprocess_text, Preprocessor
import joblib
# import lightgbm as lgb
import optuna
# import optuna.integration.lightgbm as lgb
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
    debug = config['parameters']['debug']
    configs = configs['parameters']['train_stage_2']

    if not debug:
        wandb.login(key=os.environ['WANDB'])
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            group='training_stage_2',
            config=configs)
    return wandb


def objective(trial, dtrain, dval):
    max_depth = trial.suggest_int('max_depth', 2, 10)
    params = {
        'boosting_type': 'gbdt',
        'random_state': 42,
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, ),
        'max_depth': max_depth,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 2**max_depth - 1),
        'verbosity': -1  # Add this line to suppress warnings and info messages
    }

    evaluation_results = {}
    model = lgb.train(params,
                      num_boost_round=10000,
                      valid_names=['train', 'valid'],
                      train_set=dtrain,
                      valid_sets=dval,
                    #   verbose_eval=1000,
                    #   early_stopping_rounds=30,
                      callbacks=[
                          lgb.early_stopping(
                                        stopping_rounds=30, verbose=True),
                          lgb.record_evaluation(evaluation_results)])

    # Use the last metric for early stopping
    evals_result = model.best_score
    last_metric = list(evals_result.values())[-1]
    trial.set_user_attr('best_model', model)  # Save the model in the trial
    return last_metric[list(last_metric.keys())[-1]]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_main(config):
    config_ = OmegaConf.to_yaml(config)
    # print(config_)
    cfg.__dict__.update(config.parameters)
    prompts_train, prompts_test, summary_train, summary_test, submissions = read_data(
        data_dir=cfg.root_data_dir)
    prompt_grade = read_prompt_grade(cfg.grade_data_dir)
    prompts_train = preprocess_and_join(
        prompts_train,
        prompt_grade,
        'prompt_title',
        'title',
        'grade')
    # train = prompts_train.merge(summary_train, on="prompt_id")
    # if cfg.preprocess_text:
    #     LOGGER.info("Performing preprocess text")
    #     train = preprocess_text(train)
    # train_with_output = read_data_stage_2(data_dir=cfg.train_stage_2.input_dir,file_name=cfg.train_stage_2.input_file)
    preprocessor = Preprocessor()
    # prompts_train.fillna('nan',inplace=True)
    # summary_train.fillna('nan',inplace=True)
    train = preprocessor.run(prompts_train, summary_train, mode="train")
    train["stage_1_content"] = train["content"]
    train["stage_1_wording"] = train["wording"]
    print(train['grade'].value_counts())
    train = slit_folds(
        train,
        n_fold=cfg.n_fold,
        seed=42,
        strategy=cfg.train_stage_2.strategy)
    print(train['grade'].value_counts())
    targets = ["content", "wording"]

    drop_columns = [
        "fold",
        "student_id",
        "prompt_id",
        "text",
        # "fixed_summary_text",
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

    ] + targets

    model_dict = {}

    for target in targets:
        models = []

        for fold in range(cfg.n_fold):

            X_train_cv = train[train["fold"] !=
                               fold].drop(columns=drop_columns)
            y_train_cv = train[train["fold"] != fold][target]

            X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
            y_eval_cv = train[train["fold"] == fold][target]

            dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
            dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)
            if not cfg.train_stage_2.use_optuna:
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
                                # categorical_feature = categorical_features,
                                valid_names=['train', 'valid'],
                                train_set=dtrain,
                                valid_sets=dval,
                                callbacks=[
                                    lgb.early_stopping(
                                        stopping_rounds=30, verbose=True),
                                    lgb.log_evaluation(100),
                                    lgb.callback.record_evaluation(
                                        evaluation_results)
                                ],
                                )
            else:
                study = optuna.create_study(direction='minimize')
                study.optimize(lambda trial: objective(trial, dtrain=dtrain, dval=dval), n_trials=100)
                
                print('Best trial: score {}, params {}'.format(study.best_value, study.best_params))

                model = study.trials[study.best_trial.number].user_attrs['best_model']
            # save model
            joblib.dump(
                model,
                os.path.join(
                    cfg.train_stage_2.output_model_dir,
                    cfg.train_stage_2.output_model_name.format(
                        target=target,
                        fold=fold)))
            model = joblib.load(os.path.join(
                cfg.train_stage_2.output_model_dir,
                cfg.train_stage_2.output_model_name.format(
                    target=target,
                    fold=fold)))
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
