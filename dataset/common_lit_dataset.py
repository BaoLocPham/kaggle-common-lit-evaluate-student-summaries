import torch
from torch.utils.data import Dataset, DataLoader
import re



class TrainDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.train_stage_1.max_len
        self.pt = df['prompt_title'].values
        self.pq = df['prompt_question'].values
        self.text = df['text'].values
        self.targets = df[['content', 'wording']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pt = self.pt[index]
        pq = self.pq[index]
        text = self.text[index]
        full_text = pt + " " + self.tokenizer.sep_token + \
            pq + " " + self.tokenizer.sep_token + " " + text
        # full_text = full_text.replace("\n", "|")
        # full_text = re.sub('<[^<]+?>', '', full_text)

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

        }, torch.tensor(target, dtype=torch.float)


class TestDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.inference_stage_1.max_len
        self.pt = df['prompt_title'].values
        self.pq = df['prompt_question'].values
        self.text = df['text'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pt = self.pt[index]
        pq = self.pq[index]
        text = self.text[index]
        full_text = pt + " " + self.tokenizer.sep_token + \
            pq + " " + self.tokenizer.sep_token + " " + text
        # full_text = full_text.replace("\n", "|")
        # full_text = re.sub('<[^<]+?>', '', full_text)

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
        }


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs
