import os
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
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
from datasets import Dataset, load_dataset, disable_progress_bar
import pandas as pd
import numpy as np
from configs.config_class import Eval_Config

from metrics.rcrmse import compute_mcrmse
from nlp import eval_tokenize

warnings.simplefilter("ignore")
logging.disable(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

disable_progress_bar()


def main():
    parser = HfArgumentParser((Eval_Config, TrainingArguments))

    config, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path.format(
            fold=config.folds[0]))

    if training_args.do_eval and training_args.do_predict:
        raise ValueError("Choose one of `do_eval` and `do_predict`. Not both.")

    if training_args.do_predict:
        pdf_file = f"{config.data_dir}/prompts_test.csv"
        sdf_file = f"{config.data_dir}/summaries_test.csv"
    elif training_args.do_eval:
        pdf_file = f"{config.data_dir}/prompts_train.csv"
        sdf_file = f"{config.data_dir}/summaries_train.csv"
    else:
        raise ValueError(
            "Choose `do_eval` to run validation on OOF and `do_predict` to get predictions on test set")

    pdf = pd.read_csv(pdf_file)
    sdf = pd.read_csv(sdf_file)

    df = pdf.merge(sdf, on="prompt_id")

    ds = Dataset.from_pandas(df)

    tokenized_ds_path = Path(config.tokenized_ds_path) / "tokenized.pq"

    disable_progress_bar()
    # Only need to tokenize once
    if tokenized_ds_path.exists():
        ds = load_dataset(
            "parquet",
            data_files=str(tokenized_ds_path),
            split="train")
    else:

        ds = ds.map(
            eval_tokenize,
            batched=False,
            num_proc=config.num_proc,
            fn_kwargs={"tokenizer": tokenizer, "config": config},
        )

        # sort by length to reduce padding, speed it up
        ds = ds.sort("length")

        cols2keep = ["student_id", "prompt_id"]

        if not tokenized_ds_path.exists():
            tds_dir = Path(config.tokenized_ds_path)
            tds_dir.mkdir(parents=True, exist_ok=True)

            ds.to_parquet(str(tds_dir / "tokenized.pq"))

            temp = ds.remove_columns(
                [x for x in ds.column_names if x not in cols2keep])
            temp.to_json(str(tds_dir / "ids.json"))

    if training_args.do_eval:
        id2fold = {
            "814d6b": 0,
            "39c16e": 1,
            "3b9047": 2,
            "ebad26": 3,
        }

        full_ds = ds.map(lambda x: {"fold": id2fold[x["prompt_id"]]})
        folds = [
            [i for i, f in enumerate(full_ds["fold"]) if f == fold]
            for fold in range(4)
        ]

    base_output = str(training_args.output_dir)

    for fold in config.folds.split(";"):

        if training_args.do_eval:
            ds = full_ds.select(folds[int(fold)])
        print("FOLD", fold)
        model_path = config.model_name_or_path.format(fold=fold)

        training_args_dict = vars(training_args)
        training_args_dict['output_dir'] = base_output.format(fold=fold)
        key_to_del = [
            '__cached__setup_devices',
            '_frozen',
            '_n_gpu',
            'deepspeed_plugin',
            'distributed_state']
        for key in key_to_del:
            if key in training_args_dict:
                del training_args_dict[key]
        HFParser = HfArgumentParser(TrainingArguments)
        training_args = HFParser.parse_dict(training_args_dict)[0]
        if training_args.output_dir.startswith("/kaggle/input/"):
            output_dir = training_args.output_dir[len("/kaggle/input/"):]
        if training_args.output_dir.startswith("/kaggle/working"):
            output_dir = training_args.output_dir[len("/kaggle/working"):]
        if training_args.output_dir.startswith(
                "/content/kaggle-common-lit-evaluate-student-summaries/"):
            output_dir = training_args.output_dir[len(
                "/content/kaggle-common-lit-evaluate-student-summaries/"):]
        output_dir = output_dir.replace("/", "_")

        if training_args.process_index == 0:
            print(f"Running {model_path}")

        model_config = AutoConfig.from_pretrained(model_path)
        # This is slightly faster than doing `from_pretrained`
        model = AutoModelForSequenceClassification.from_config(model_config)
        model.load_state_dict(torch.load(model_path + "/pytorch_model.bin"))

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=compute_mcrmse,
        )
        os.makedirs(output_dir, exist_ok=True)
        if training_args.do_predict:
            predictions = trainer.predict(ds).predictions

            np.save(os.path.join(output_dir, f"predictions.npy"), predictions)

        else:
            predictions = trainer.predict(ds)
            metrics = predictions.metrics
            trainer.save_metrics("eval", metrics)
            if training_args.process_index == 0:
                print(metrics)
                print(
                    f"best mcrmse during training: {model.config.best_metric}")

            np.save(
                os.path.join(
                    output_dir,
                    f"predictions.npy"),
                predictions.predictions)


if __name__ == "__main__":
    main()
