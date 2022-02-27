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
from transformers import AutoConfig
from transformers import AutoModel
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import Trainer

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
    
## Test Dataset
class SarcasmTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    def __len__(self):
        return len(self.encodings)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.3]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred)

    return {"accuracy": accuracy,"f1_score":f1}

def labels(x):
    if x == 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    path = '../../Data/Train_Dataset.csv'
    path_test = '../../Data/Test_Dataset.csv'

    df = pd.read_csv(path)
    test = pd.read_csv(path_test)
    df = df.dropna(subset=['tweet'])

    train = df

    train_tweets = train['tweet'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['text'].values.tolist()

    train_tweets, val_tweets, train_labels, val_labels = train_test_split(train_tweets, train_labels, 
                                                                        test_size=0.1,random_state=42,stratify=train_labels)

    model_name = 'detecting-Sarcasm'

    tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-cased',
                                                        num_labels=2,loss_function_params={"weight": [0.75, 0.25]})

    train_encodings = tokenizer(train_tweets, truncation=True, padding=True,return_tensors = 'pt')
    val_encodings = tokenizer(val_tweets, truncation=True, padding=True,return_tensors = 'pt')
    test_encodings = tokenizer(test_tweets, truncation=True, padding=True,return_tensors = 'pt')

    train_dataset = SarcasmDataset(train_encodings, train_labels)
    val_dataset = SarcasmDataset(val_encodings, val_labels)
    test_dataset = SarcasmTestDataset(test_encodings)

    training_args = TrainingArguments(
        output_dir='./res', evaluation_strategy="steps", num_train_epochs=5, per_device_train_batch_size=32,
        per_device_eval_batch_size=64, warmup_steps=500, weight_decay=0.01,logging_dir='./logs4',
        load_best_model_at_end=True,
    )

    model = DistilBertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=2)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

    test['sarcastic'] = 0
    test_tweets = test['tweet'].values.tolist() 
    test_labels = test['sarcastic'].values.tolist() 
    test_encodings = tokenizer(test_tweets,truncation=True, 
                            padding=True,return_tensors = 'pt').to("cuda") 

    preds = trainer.predict(test_dataset=test_dataset)

    probs = torch.from_numpy(preds[0]).softmax(1)
    predictions = probs.numpy()

    newdf = pd.DataFrame(predictions,columns=['Negative_1','Positive_2'])

    results = np.argmax(predictions,axis=1)

    model.predict(test_dataset) 