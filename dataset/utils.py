import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold


def read_data(data_dir: str):
    prompts_train = pd.read_csv(os.path.join(data_dir, 'prompts_train.csv'))
    prompts_test = pd.read_csv(os.path.join(data_dir, 'prompts_test.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    summary_train = pd.read_csv(os.path.join(data_dir, 'summaries_train.csv'))
    summary_test = pd.read_csv(os.path.join(data_dir, 'summaries_test.csv'))
    return prompts_train, prompts_test,  summary_train, summary_test, submission

def read_test(data_dir: str):
    prompts_test = pd.read_csv(os.path.join(data_dir, 'prompts_test.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    summary_test = pd.read_csv(os.path.join(data_dir, 'summaries_test.csv'))
    return prompts_test, summary_test, submission

def merge_prompt_summary(prompts, summary):
    return prompts.merge(summary, on="prompt_id")


def slit_folds(train: pd.DataFrame, n_fold, seed, strategy ="GroupKFold"):
    train['fold'] = -1
    if strategy =="GroupKFold":
        fold = GroupKFold(n_splits=n_fold)
        for n, (train_index, val_index) in enumerate(fold.split(train, groups=train['prompt_id'])):
            train.loc[val_index, 'fold'] = n
        train['fold'] = train['fold'].astype(int)
        fold_sizes = train.groupby('fold').size()
        return train
    
    elif "StratifiedKFold":
        fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
        for n, (train_index, val_index) in enumerate(fold.split(train, train['prompt_id'])):
            train.loc[val_index, 'fold'] = n
        train['fold'] = train['fold'].astype(int)
        fold_sizes = train.groupby('fold').size()
        return train
    elif "StratifiedKFold_bin10":
        train["bin10_content"] = pd.cut(train["content"], bins=10, labels=list(range(10)))
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for n, (train_index, valid_index) in enumerate(fold.split(train, train["bin10_content"])):
            train.loc[valid_index, "fold"] = n
        return train
