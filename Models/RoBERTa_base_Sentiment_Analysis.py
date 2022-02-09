import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer,TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


class SarcasimDataset(torch.utils.data.Dataset):
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
class SarcasimTestDataset(torch.utils.data.Dataset):
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
    #recall = recall_score(y_true=labels, y_pred=pred)
    #precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(labels, pred, average='weighted')

    return {"accuracy": accuracy,"f1_score":f1}

def labels(x):
    if x == 0:
        return 0
    else:
        return 1

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

if __name__ == '__main__':
    #dataset address
    dataset_path = '../Data/English/train.En.csv'
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['tweet'])

    train, test = train_test_split(df, test_size=0.1)

    train_tweets = train['tweet'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['tweet'].values.tolist()
    test_labels = test['sarcastic']

    train_tweets, val_tweets, train_labels, val_labels = train_test_split(train_tweets, train_labels, 
                                                                        test_size=0.1,random_state=42,stratify=train_labels)

    model_name = 'detecting-sarcasim'

    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL,
                                            num_labels=2,
                                            loss_function_params={"weight": [0.75, 0.25]}
                                                        )
    train_encodings = tokenizer(train_tweets, truncation=True, padding=True,return_tensors = 'pt')
    val_encodings = tokenizer(val_tweets, truncation=True, padding=True,return_tensors = 'pt')
    test_encodings = tokenizer(test_tweets, truncation=True, padding=True,return_tensors = 'pt')


    train_dataset = SarcasimDataset(train_encodings, train_labels)
    val_dataset = SarcasimDataset(val_encodings, val_labels)
    test_dataset = SarcasimTestDataset(test_encodings)

    training_args = TrainingArguments(
        output_dir='./res', evaluation_strategy="steps", num_train_epochs=5, per_device_train_batch_size=32,
        per_device_eval_batch_size=64, warmup_steps=500, weight_decay=0.01,logging_dir='./logs4',
        #logging_steps=10,
        load_best_model_at_end=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    model.save_pretrained(MODEL)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()


    #TEST

    pin_memory=False
    preds = trainer.predict(test_dataset=test_dataset)
    probs = torch.from_numpy(preds[0]).softmax(1)

    # convert tensors to numpy array
    predictions = probs.numpy()

    newdf = pd.DataFrame(predictions,columns=['Negative_1','Positive_2'])



    results = np.argmax(predictions,axis=1)
    test['sarcastic_result'] =  test['sarcastic'].map(labels)

    print(f1_score(test_labels, test['sarcastic_result']))

    model.predict(test['tweet'])