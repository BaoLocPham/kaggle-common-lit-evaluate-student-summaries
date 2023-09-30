import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import re
import string
import spacy

def preprocess_text_helper(text):
    try:
        text = text.replace("\n", " ")
        words = text.split()
        words = [word.lower() for word in words]
        table = str.maketrans("", "", string.punctuation)
        stripped = [w.translate(table) for w in words]
        words = [word for word in stripped if word.isalpha()]
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        words = [w for w in words if not w in stop_words]
        text = " ".join(words)
        return text
    except:
        return text
    
def preprocess_text(data: pd.DataFrame):
    data["prompt_title"] = data["prompt_title"].apply(lambda x: preprocess_text_helper(x))
    data["prompt_question"] = data["prompt_question"].apply(
        lambda x: preprocess_text_helper(x)
    )
    data["prompt_text"] = data["prompt_text"].apply(lambda x: preprocess_text_helper(x))
    data["text"] = data["text"].apply(lambda x: preprocess_text_helper(x))
    return data

def read_data(data_dir: str):
    prompts_train = pd.read_csv(os.path.join(data_dir, 'prompts_train.csv'))
    prompts_test = pd.read_csv(os.path.join(data_dir, 'prompts_test.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    summary_train = pd.read_csv(os.path.join(data_dir, 'summaries_train.csv'))
    summary_test = pd.read_csv(os.path.join(data_dir, 'summaries_test.csv'))
    return prompts_train, prompts_test,  summary_train, summary_test, submission

def read_data_stage_2(data_dir, file_name):
    data_df = pd.read_csv(os.path.join(data_dir, file_name))
    return data_df

def read_submission(data_dir: str):
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    return submission

def read_test(data_dir: str):
    prompts_test = pd.read_csv(os.path.join(data_dir, 'prompts_test.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    summary_test = pd.read_csv(os.path.join(data_dir, 'summaries_test.csv'))
    return prompts_test, summary_test, submission

def merge_prompt_summary(prompts, summary):
    return prompts.merge(summary, on="prompt_id")

def read_prompt_grade(data_dir: str):
    prompt_grade = pd.read_csv(os.path.join(data_dir, 'prompt_grade.csv'))
    return prompt_grade

def preprocess_and_join(df1, df2, df1_title_col, df2_title_col, grade_col):
    # Copy dataframes to avoid modifying the originals
    df1 = df1.copy()
    df2 = df2.copy()

    # Preprocess titles
    df1[df1_title_col] = df1[df1_title_col].str.replace('"', '').str.strip()
    df2[df2_title_col] = df2[df2_title_col].str.replace('"', '').str.strip()

    # Remove duplicate grades
    df2 = df2.drop_duplicates(subset=df2_title_col, keep='first')

    # Join dataframes
    merged_df = df1.merge(df2, how='left', left_on=df1_title_col, right_on=df2_title_col)
    

    # Postprocess grades
    merged_df[grade_col] = merged_df[grade_col].fillna(0)
    merged_df[grade_col] = merged_df[grade_col].astype(int).astype('category')

    return merged_df

def slit_folds(train: pd.DataFrame, n_fold, seed, strategy ="GroupKFold"):
    train['fold'] = -1
    if n_fold==1:
        train["fold"] = 0
        return train

    print(f"Splitting by strategy: {strategy}")
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
    elif "GroupKFold_grade":
        fold = GroupKFold(n_splits=n_fold)
        for n, (train_index, val_index) in enumerate(fold.split(train, groups=train['grade'])):
            train.loc[val_index, 'fold'] = n
        train['fold'] = train['fold'].astype(int)
        fold_sizes = train.groupby('fold').size()
        return train
