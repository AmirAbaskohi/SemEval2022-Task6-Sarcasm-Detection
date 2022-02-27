import torch.nn as nn
import numpy as np
import pandas as pd
import os
import re
import json
import copy
import collections
import time
import pickle
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.utils import shuffle
from numpy.lib.function_base import average
from tqdm.notebook import tqdm
from collections import Counter
from transformers import BertConfig, BertTokenizer, BertweetTokenizer, RobertaTokenizer, AlbertTokenizer, DistilBertTokenizer, XLMRobertaTokenizer, XLNetTokenizer, T5Tokenizer
from transformers import BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from transformers import AutoTokenizer, XLMRobertaTokenizer
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, DistilBertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification, XLMRobertaForSequenceClassification, XLNetForSequenceClassification, T5Model
from transformers import TrainingArguments
from transformers import Trainer
from google.colab import drive

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class SarcasmTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)

    return {"accuracy": accuracy,"f1_score":f1}


if __name__ == '__main__':
    train = pd.read_csv('../../Data/Train_Dataset.csv')
    test = pd.read_csv('../../Data/Test_Dataset.csv')

    X_train = train['tweet']
    y_train = train['sarcastic']
    X_test = train['text']
    y_test = train['sarcastic']

    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()

    model_name = 'detecting-sarcasim'
    task='sentiment'
    MODEL = 'xlnet-base-cased'

    tokenizer = XLNetTokenizer.from_pretrained(MODEL,num_labels=2, loss_function_params={"weight": [0.75, 0.25]})

    train_encodings = tokenizer(X_train, truncation=True, padding=True,return_tensors = 'pt')
    test_encodings = tokenizer(X_test,truncation=True, padding=True,return_tensors = 'pt')

    train_dataset = SarcasimDataset(train_encodings, y_train)
    test_dataset = SarcasimDataset(test_encodings, y_test)

    training_args = TrainingArguments(
        output_dir='./res', num_train_epochs=5, per_device_train_batch_size=32, warmup_steps=500, weight_decay=0.01,logging_dir='./logs4'
    )

    model = XLNetForSequenceClassification.from_pretrained(MODEL)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        compute_metrics = compute_metrics,
    )
    trainer.train()

    preds = trainer.predict(test_dataset)
    preds = np.argmax(preds.predictions[:, 0:2], axis=-1)
    print(f1_score(y_test, preds))