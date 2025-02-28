from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import MT5Tokenizer


class MyDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: MT5Tokenizer, classes_map: Dict, prefix_text: str):
        self.data = data
        self.tokenizer = tokenizer

        self.labels_map = {value: key for key, value in classes_map.items()}  # labels_map = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}
        self.prefix_text = prefix_text

    def __getitem__(self, index):
        text = self.data[index]["text"]
        label = self.data[index]["label"]

        # 编码输入文本
        text_encoded = self.tokenizer.encode_plus(
            text=self.prefix_text + text,
            add_special_tokens=True,  # T5没有[CLS]和[SEP]，有结束符</s>
            return_attention_mask=True,
            max_length=512,  # 设置最大长度
            padding="max_length",  # 填充到最大长度
            truncation=True  # 截断过长的文本
        )

        # 编码标签
        labels_id = self.tokenizer.encode(
            text=self.labels_map[label],
            add_special_tokens=False  # 不添加特殊标记，因为 mt5 会自动添加 </s>
        )

        return {
            "input_ids": torch.tensor(text_encoded["input_ids"]),
            "attention_mask": torch.tensor(text_encoded["attention_mask"]),
            "labels": torch.tensor(labels_id)
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids_list = [instance['input_ids'] for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask_list = [instance['attention_mask'] for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        labels_list = [instance['labels'] for instance in batch]
        labels_pad = pad_sequence(labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids_pad,
            "attention_mask": attention_mask_pad,
            "labels": labels_pad
        }
