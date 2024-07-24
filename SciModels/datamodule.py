import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TextDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_seq_length,
        output_hidden_states=False,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_length = max_seq_length
        self.output_hidden_states = output_hidden_states

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data[index]
        data_row["id"] = index

        return data_row

    def collate_fn(self, batch):
        input_text, labels, ids = [], [], []

        for instance in batch:
 
            input_text.append(instance["text"])
            labels.append(instance["label"])
            ids.append(instance["index"])

        text_encoding = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = torch.LongTensor(labels)

        if self.output_hidden_states:
            return dict(
                ids=ids,
                text_input_ids=text_encoding['input_ids'],
                text_attention_mask=text_encoding['attention_mask'],
                labels=labels,
            )
        else:
            return dict(
                text_input_ids=text_encoding['input_ids'],
                text_attention_mask=text_encoding['attention_mask'],
                labels=labels,
            )


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        tokenizer,
        batch_size,
        max_seq_length,
        output_hidden_states=False,
    ):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.output_hidden_states = output_hidden_states

    def setup(self, stage=None):
        self.train_dataset = TextDataset(
            self.train_data,
            self.tokenizer,
            self.max_seq_length,
            self.output_hidden_states
        )

        if self.val_data is not None:
            self.val_dataset = TextDataset(
                self.val_data,
                self.tokenizer,
                self.max_seq_length,
                self.output_hidden_states
            )
        else:
            self.val_dataset = None

        if self.test_data is not None:
            self.test_dataset = TextDataset(
                self.test_data,
                self.tokenizer,
                self.max_seq_length,
                self.output_hidden_states
            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
        )
    
