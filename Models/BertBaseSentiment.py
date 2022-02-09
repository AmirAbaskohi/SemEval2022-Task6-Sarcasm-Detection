import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import Trainer,TrainingArguments
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer

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
    f1 = f1_score(labels, pred, average='weighted')

    return {"accuracy": accuracy,"f1_score":f1}

def labels(x):
    if x == 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    path = '../Data/Train_Dataset.csv'
    path_test = '../Data/Test_Dataset.csv'

    df = pd.read_csv(path)
    test = pd.read_csv(path_test)
    df = df.dropna(subset=['tweet'])

    train = df

    train_tweets = train['tweet'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['text'].values.tolist()

    train_tweets, val_tweets, train_labels, val_labels = train_test_split(train_tweets, train_labels, 
                                                                        test_size=0.1,random_state=42,stratify=train_labels)
    model_name = 'detecting-sarcasim'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', num_labels=2,
                                            loss_function_params={"weight": [0.75, 0.25]})


    train_encodings = tokenizer(train_tweets, padding=True, truncation=True, max_length=512)
    val_encodings = tokenizer(val_tweets, padding=True, truncation=True, max_length=512)
    test_encodings = tokenizer(test_tweets, padding=True, truncation=True, max_length=512)

    train_dataset = SarcasimDataset(train_encodings, train_labels)
    val_dataset = SarcasimDataset(val_encodings, val_labels)
    test_dataset = SarcasimTestDataset(test_encodings)

    training_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=0,
        load_best_model_at_end=True,
    )

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

    preds = trainer.predict(test_dataset=test_dataset)

    probs = torch.from_numpy(preds[0]).softmax(1)

    predictions = probs.numpy()

    newdf = pd.DataFrame(predictions,columns=['Negative_1','Positive_2'])

    results = np.argmax(predictions,axis=1)

    test['sarcastic'] = 0
    test_tweets = test['tweet'].values.tolist() 
    test_labels = test['sarcastic'].values.tolist() 
    test_encodings = tokenizer(test_tweets,
                            truncation=True, 
                            padding=True,
                            return_tensors = 'pt').to("cuda") 

    f1_score(test_labels, test['sarcastic_result'])
