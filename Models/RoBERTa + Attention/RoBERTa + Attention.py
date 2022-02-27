import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from transformers import Trainer,TrainingArguments
from transformers import DistilBertTokenizerFast, BertForMaskedLM
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from google.colab import drive
from torch import nn
from transformers import Trainer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import time
from transformers import AutoModel
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

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

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      self.dropout = nn.Dropout(0.1)
      self.relu =  nn.ReLU()
      self.fc1 = nn.Linear(768,512)
      self.fc2 = nn.Linear(512,2)
      self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)
      return x

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)

    return {"accuracy": accuracy,"f1_score":f1}

def train():
  model.train()
  total_loss, total_accuracy = 0, 0
  total_preds=[]
  for step,batch in enumerate(train_dataloader):
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    sent_id, mask, labels = batch
    model.zero_grad()        
    preds = model(sent_id, mask)
    loss = cross_entropy(preds, labels)
    total_loss = total_loss + loss.item()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    preds=preds.detach().cpu().numpy()

    total_preds.append(preds)

  avg_loss = total_loss / len(train_dataloader)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

def evaluate():
  print("\nEvaluating...")
  model.eval()

  total_loss, total_accuracy = 0, 0
  total_preds = []
  for step,batch in enumerate(test_dataloader):
    if step % 50 == 0 and not step == 0:
      elapsed = format_time(time.time() - t0)
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    sent_id, mask, labels = batch
    with torch.no_grad():
      preds = model(sent_id, mask)
      loss = cross_entropy(preds,labels)
      total_loss = total_loss + loss.item()
      preds = preds.detach().cpu().numpy()
      total_preds.append(preds)

  avg_loss = total_loss / len(test_dataloader) 
  total_preds  = np.concatenate(total_preds, axis=0)
  return avg_loss, total_preds

if __name__ == '__main__':
    train = pd.read_csv('../../Data/Train_Dataset.csv')
    test = pd.read_csv('../../Data/Test_Dataset.csv')

    X_train = train['tweet']
    y_train = train['sarcastic']
    X_test = test['text']
    y_test = test['sarcastic']

    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()

    model_name = 'detecting-sarcasim'
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL,num_labels=2, loss_function_params={"weight": [0.75, 0.25]})

    model = AutoModel.from_pretrained(MODEL)

    train_encodings = tokenizer(X_train, truncation=True, padding=True,return_tensors = 'pt')
    test_encodings = tokenizer(X_test,truncation=True, padding=True,return_tensors = 'pt')

    train_dataset = SarcasimDataset(train_encodings, y_train)
    test_dataset = SarcasimDataset(test_encodings, y_test)

    training_args = TrainingArguments(
        output_dir='./res', num_train_epochs=5, per_device_train_batch_size=32, warmup_steps=500, weight_decay=0.01,logging_dir='./logs4'
    )

    seq_len = [len(i.split()) for i in X_train]

    tokens_train = tokenizer.batch_encode_plus(
        X_train,
        max_length = 25,
        pad_to_max_length=True,
        truncation=True
    )
    train_seq = torch.tensor(tokens_train['input_ids'])

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(y_train)

    tokens_test = tokenizer.batch_encode_plus(
        X_test,
        max_length = 25,
        pad_to_max_length=True,
        truncation=True
    )

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(y_test)

    batch_size = 32
    train_data = TensorDataset(train_seq, train_mask, train_y)

    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_seq, test_mask, test_y)
    batch_size = 32
    
    test_sampler = SequentialSampler(test_data)

    test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)

    for param in model.parameters():
        param.requires_grad = False

    model = BERT_Arch(model)

    optimizer = AdamW(model.parameters(), lr = 1e-5) 

    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(train['sarcastic']), y = train['sarcastic'])
    
    weights= torch.tensor(class_weights,dtype=torch.float)

    cross_entropy  = nn.NLLLoss(weight=weights) 
    epochs = 5
    best_valid_loss = float('inf')

    train_losses=[]
    valid_losses=[]
    for epoch in range(epochs):  
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        train_loss, _ = train()
        valid_loss, _ = evaluate()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    path = 'saved_weights.pt'
    model.load_state_dict(torch.load(path))

    with torch.no_grad():
        preds = model(test_seq, test_mask)
        preds = preds.detach().numpy()

    preds = np.argmax(preds, axis = 1)
    print(classification_report(test_y, preds))

    print(f1_score(test_y, preds))