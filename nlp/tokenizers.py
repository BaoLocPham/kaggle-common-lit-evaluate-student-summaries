from typing import Any


def tokenize(example, tokenizer, config):
    sep = tokenizer.sep_token

    cols = []
    if config.add_prompt_title:
        cols.append("prompt_title")
    elif config.add_prompt_question:
        cols.append("prompt_question")
    elif config.add_prompt_text:
        cols.append("prompt_text")
    cols.append("text")

    labels = [example["content"], example["wording"]]

    tokenized = tokenizer(
        sep.join([example[c] for c in cols]),
        padding=False,
        truncation=True,
        max_length=config.max_seq_length,
    )

    return {
        **tokenized,
        "labels": labels,
    }

def eval_tokenize(example, tokenizer, config):
    sep = tokenizer.sep_token


    cols = []

    if config.add_prompt_title:
        cols.append("prompt_title")
    elif config.add_prompt_question:
        cols.append("prompt_question")
    elif config.add_prompt_text:
        cols.append("prompt_text")
        
    cols.append("text")

    text = sep.join([example[col] for col in cols])

    

    tokenized = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=config.max_seq_length,
    )

    if "content" in example:
        tokenized["labels"] = [example["content"], example["wording"]]

    tokenized["length"] = len(tokenized["input_ids"])

    return {
        **tokenized,
    }