import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import AdamW
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from typing import Optional

from transformers import ElectraModel, ElectraPreTrainedModel, ElectraTokenizerFast as ElectraTokenizer
from transformers.models.electra.modeling_electra import ElectraClassificationHead

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

pl.seed_everything(42)

class TweetDataset(Dataset):
    def __init__(self, data:pd.DataFrame, tokenizer: ElectraTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row.tweet,
            max_length = 64,
            truncation = True,
            padding = "max_length",
            add_special_tokens = True,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors = "pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(row.sarcastic)
        }

class TweetDataModule(pl.LightningDataModule):
    def __init__(self, data:pd.DataFrame, tokenizer: ElectraTokenizer, batch_size: int):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.setup()
    
    def setup(self, stage: Optional[str] = None):
        ### add for loading 
        train_df = pd.read_csv('/content/drive/MyDrive/isarcasm/isarcasm_datasets/Train_Dataset.csv')[['tweet', 'sarcastic']]
        train_df = train_df[train_df['tweet'].notna()]
        self.train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        test_df = pd.read_csv('/content/drive/MyDrive/isarcasm/isarcasm_datasets/Test_Dataset.csv')[['tweet', 'sarcastic']]
        test_df = test_df[test_df['tweet'].notna()]
        self.test_df = test_df

        self.train_df, self.val_df = train_test_split(self.train_df, test_size=0.1)
    
    def train_dataloader(self):
        return DataLoader(
            dataset = TweetDataset(self.train_df, self.tokenizer),
            batch_size = self.batch_size,
            num_workers = os.cpu_count(),
            shuffle = True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset = TweetDataset(self.val_df, self.tokenizer),
            batch_size = self.batch_size,
            num_workers = os.cpu_count(),
            shuffle = False
        )  

    def test_dataloader(self):
        return DataLoader(
            dataset = TweetDataset(self.test_df, self.tokenizer),
            batch_size = self.batch_size,
            num_workers = os.cpu_count(),
            shuffle = False
        )

class ElectraClassifier(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_classes = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

        self.post_init()

    def forward(self, input_ids = None, attention_mask = None):
        discriminator_hidden_states = self.electra(input_ids, attention_mask)
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        return logits

class SarcasmClassifier(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.classifier = ElectraClassifier.from_pretrained("google/electra-small-discriminator", num_labels = n_classes)

        class_weights = torch.FloatTensor([1, 3]).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.logits = None
        self.preds = []
    
    def forward(self, input_ids, attention_mask):
        return self.classifier(input_ids, attention_mask)
    
    def run_step(self, batch, stage):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].long()
        logits = self(input_ids, attention_mask)

        self.logits = logits

        loss = self.criterion(logits, labels)

        # accuracy = torchmetrics.Accuracy()(logits, labels)

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.run_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        r = self.run_step(batch, "test")
        self.preds += list(self.logits.cpu().data.numpy().argmax(axis=1))
        return r

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4)
    
if __name__ == "__main__":
    df = pd.read_csv('../Data/Train_Dataset.csv')[['tweet', 'sarcastic']]
    df = df[df['tweet'].notna()]
    
    MODEL_NAME = "google/electra-base-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
    
    data_module = TweetDataModule(df, tokenizer, batch_size = 32)
    
    clf = SarcasmClassifier(2)
    trainer = Trainer(max_epochs=20, gpus=1, accelerator="gpu", log_every_n_steps=1)

    trainer.fit(clf, data_module.train_dataloader(), data_module.val_dataloader())
    
    trainer.test(clf, data_module.test_dataloader())
    
    y_pred = clf.preds

    y_test = []
    for i in data_module.test_dataloader().dataset:
        y_test.append(i['label'].data.numpy())

    print(f1_score(y_test, y_pred))