from copy import copy
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

class SarcasmDataset(Dataset):
  def __init__(self, tokenizer, df, max_len=512):
    self.max_len = max_len
    self.df = df
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()
    target_mask = self.targets[index]["attention_mask"].squeeze()

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    for index, row in self.df.iterrows():
      
      line = copy(row['tweet'])
      
      target = ""
      if row['sarcastic'] == 1:
        target = "positive"
      else:
        target = "negative"

      line = line.strip()
      line = line + ' </s>'

      target = target + " </s>"

      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )

      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
      )

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)