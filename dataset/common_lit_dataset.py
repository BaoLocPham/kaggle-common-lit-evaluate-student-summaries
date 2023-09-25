import torch
from torch.utils.data import Dataset, DataLoader
import re



class TrainDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.train_stage_1.max_len
        self.max_len_char_title = cfg.train_stage_1.max_len_char_title
        self.max_len_char_question = cfg.train_stage_1.max_len_char_question
        self.max_len_char_prompt_text = cfg.train_stage_1.max_len_char_prompt_text
        self.full_text = cfg.train_stage_1.full_text
        self.pt = df['prompt_title'].values
        self.pq = df['prompt_question'].values
        self.ptext = df['prompt_text'].values
        self.text = df['fixed_summary_text'].values
        self.targets = df[['content', 'wording']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pt = self.pt[index][:self.max_len_char_title]
        pq = self.pq[index][:self.max_len_char_question]
        ptext = self.ptext[index][:self.max_len_char_prompt_text]
        text = self.text[index]
        # full_text = pt + self.tokenizer.sep_token + \
        #     pq + self.tokenizer.sep_token + ptext + \
        #         self.tokenizer.sep_token +  text

        full_text = ""
        for t in self.full_text:
            if t == "title":
                full_text += pt
            elif t == "question":
                full_text += " " + self.tokenizer.sep_token + pq
            elif t == "prompt-text":
                full_text += " " + self.tokenizer.sep_token + ptext
            elif t == "text":
                full_text += " " + self.tokenizer.sep_token + text
        # print(f"full_text: {full_text}")
        # full_text = f"{self.tokenizer.sep_token}".join(full_text)

        inputs = self.tokenizer.encode_plus(
            full_text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'

        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        target = self.targets[index]

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            # 'full_text': full_text


        }, torch.tensor(target, dtype=torch.float)


class TestDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.inference_stage_1.max_len
        self.max_len_char_title = cfg.inference_stage_1.max_len_char_title
        self.max_len_char_question = cfg.inference_stage_1.max_len_char_question
        self.max_len_char_prompt_text = cfg.inference_stage_1.max_len_char_prompt_text
        self.full_text = cfg.inference_stage_1.full_text
        self.pt = df['prompt_title'].values
        self.pq = df['prompt_question'].values
        self.ptext = df['prompt_text'].values
        self.text = df['fixed_summary_text'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pt = self.pt[index][:self.max_len_char_title]
        pq = self.pq[index][:self.max_len_char_question]
        ptext = self.ptext[index][:self.max_len_char_prompt_text]
        text = self.text[index]
        # full_text = pt + self.tokenizer.sep_token + \
        #     pq + self.tokenizer.sep_token + ptext + \
        #         self.tokenizer.sep_token +  text
        full_text = ""
        for t in self.full_text:
            if t == "title":
                full_text += pt
            elif t == "question":
                full_text += self.tokenizer.sep_token + pq
            elif t == "prompt-text":
                full_text += self.tokenizer.sep_token + ptext
            elif t == "text":
                full_text += self.tokenizer.sep_token + text
        # print(f"full_text: {text}")
        # full_text = f"{self.tokenizer.sep_token}".join(full_text)

        inputs = self.tokenizer.encode_plus(
            full_text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'

        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            # 'full_text': full_text
        }


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs
