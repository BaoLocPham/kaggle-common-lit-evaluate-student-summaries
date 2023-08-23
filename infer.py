import os
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    set_seed,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding,
)
from datasets import Dataset, disable_progress_bar
import pandas as pd
import numpy as np
from configs.config_class import Eval_Config
from metrics.rcrmse import compute_mcrmse
from nlp import tokenize

warnings.simplefilter("ignore")
logging.disable(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

disable_progress_bar()


def main():
    parser = HfArgumentParser((Eval_Config, TrainingArguments))

    config, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    if "wandb" in training_args.report_to:
        import wandb

        try:
            if os.path.exists("/kaggle/working"):
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                key = user_secrets.get_secret("wandb")

                wandb.login(key=key)
            else:
                wandb.login(key=os.environ['WANDB'])
        except:
            print("Could not log in to WandB")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    model_config = AutoConfig.from_pretrained(config.model_name_or_path)
    model_config.update({
        "hidden_dropout_prob": config.dropout,
        "attention_probs_dropout_prob": config.dropout,
        "num_labels": 2,
        "problem_type": "regression",
        "cfg": config.__dict__,
    })

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path, config=model_config, ignore_mismatched_sizes=True
    )

    pdf = pd.read_csv(f"{config.data_dir}/prompts_train.csv")
    sdf = pd.read_csv(f"{config.data_dir}/summaries_train.csv")

    df = pdf.merge(sdf, on="prompt_id")

    # 4 prompt ids, 4 folds
    id2fold = {
        "814d6b": 0,
        "39c16e": 1,
        "3b9047": 2,
        "ebad26": 3,
    }

    df["fold"] = df["prompt_id"].map(id2fold)

    train_ds = Dataset.from_pandas(df[df["fold"] != config.fold])
    val_ds = Dataset.from_pandas(df[df["fold"] == config.fold])

    train_ds = train_ds.map(
        tokenize,
        batched=False,
        num_proc=config.num_proc,
        fn_kwargs={"tokenizer": tokenizer, "config": config},
    )

    val_ds = val_ds.map(
        tokenize,
        batched=False,
        num_proc=config.num_proc,
        fn_kwargs={"tokenizer": tokenizer, "config": config},
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=16 if training_args.fp16 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_mcrmse,
    )

    trainer.train()

    model.config.best_metric = trainer.state.best_metric
    model.config.save_pretrained(training_args.output_dir)

    # need to load best model before doing predictions
#     preds = trainer.predict(val_ds).predictions

#     np.save(Path(training_args.output_dir)/f"preds_fold{config.fold}.npy", preds)

    trainer.log({"eval_best_mcrmse": trainer.state.best_metric})
    
if __name__ == "__main__":
    main()
