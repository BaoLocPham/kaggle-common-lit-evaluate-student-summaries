'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-03 16:44:52
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-14 21:14:25
 # @ Description:
    scripts for config class
'''


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    model_name_or_path: Optional[str] = field(
        default="microsoft/deberta-v3-base",
        metadata={"help": "Model name or path"},
    )

    data_dir: Optional[str] = field(
        default="/kaggle/input/commonlit-evaluate-student-summaries",
        metadata={"help": "Data directory"},
    )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max sequence length"},
    )

    add_prompt_question: Optional[bool] = field(
        default=False,
        metadata={"help": "Add prompt question into input"},
    )

    add_prompt_text: Optional[bool] = field(
        default=False,
        metadata={"help": "Add prompt text into input"},
    )

    fold: Optional[int] = field(
        default=0,
        metadata={"help": "Fold"},
    )

    num_proc: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes"},
    )

    dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Amount of dropout to apply"},
    )


@dataclass
class Eval_Config:

    model_name_or_path: Optional[str] = field(
        default="microsoft/deberta-v3-base",
        metadata={"help": "Model name or path"},
    )

    data_dir: Optional[str] = field(
        default="/kaggle/input/commonlit-evaluate-student-summaries",
        metadata={"help": "Data directory"},
    )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max sequence length"},
    )

    add_prompt_question: Optional[bool] = field(
        default=False,
        metadata={"help": "Add prompt question into input"},
    )

    add_prompt_text: Optional[bool] = field(
        default=False,
        metadata={"help": "Add prompt text into input"},
    )

    folds: Optional[str] = field(
        default="0",
        metadata={"help": "Fold"},
    )

    num_proc: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes"},
    )

    tokenized_ds_path: Optional[str] = field(
        default="tokenized_ds",
        metadata={"help": "Tokenized dataset path"},
    )
    
    overwrite_tokenized_ds: Optional[bool] = field(	
        default=False,	
        metadata={"help": "Overwrite existing tokenized datasets"},	
    )