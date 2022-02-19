import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import gc
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, df):
        super().__init__()

        self.data_list = []
        self.end_of_text_token = " <|endoftext|> "

        for index, row in df.iterrows():
            data_str = f"{row[0]}{self.end_of_text_token}"
            self.data_list.append(data_str)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]

def train(epochs, data_loader, batch_size, tokenizer, model, device):	
    batch_counter = 0
    sum_loss = 0.0

    for epoch in range(epochs):
        print (f'Running {epoch+1} epoch')
        
        for idx, txt in enumerate(data_loader):
            txt = torch.tensor(tokenizer.encode(txt[0]))
            txt = txt.unsqueeze(0).to(device)
            outputs = model(txt, labels=txt)
            loss, _ = outputs[:2]
            loss.backward()
            sum_loss += loss.data

            if idx%batch_size==0:
                batch_counter += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if batch_counter == 10:
                print(f"Total Loss is {sum_loss}")
                batch_counter = 0
                sum_loss = 0.0

    return model

def save_model(model, name):
    torch.save(model.state_dict(), f"{name}.pt")

def choose_from_top_k_top_n(probs, k=50, p=0.8):
    ind = np.argpartition(probs, -k)[-k:]
    top_prob = probs[ind]
    top_prob = {i: top_prob[idx] for idx,i in enumerate(ind)}
    sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}

    t=0
    f=[]
    pr = []
    for k,v in sorted_top_prob.items():
        t+=v
        f.append(k)
        pr.append(v)
        if t>=p:
            break
    top_prob = pr / np.sum(pr)
    token_id = np.random.choice(f, 1, p = top_prob)

    return int(token_id)

def generate(tokenizer, model, sentences, label):
    result = []
    with torch.no_grad():
        for idx in tqdm(range(sentences)):
            finished = False
            cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to(device)
            
            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]

                softmax_logits = torch.softmax(logits[0,-1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to(device).cpu().numpy())
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    finished = True
                    break

            if finished:	          
                output_list = list(cur_ids.squeeze().to(device).cpu().numpy())
                output_text = tokenizer.decode(output_list)
                result.append(output_text)
            else:
                output_list = list(cur_ids.squeeze().to(device).cpu().numpy())
                output_text = tokenizer.decode(output_list)
                result.append(output_text)
    return result

if __name__ == '__main__':
    #dataset address
    dataset_path = '../Data/Train_Dataset.csv'

    df = pd.read_csv(dataset_path)
    sarcastic = df[df['sarcastic'] == 1]
    non_sarcastic = df[df['sarcastic'] == 0]

    dataset_sarcastic = MyDataset(sarcastic)
    dataset_non_sarcastic = MyDataset(non_sarcastic)
    data_loader_sarcastic = DataLoader(dataset_sarcastic, batch_size=1, shuffle=True)
    data_loader_non_sarcastic = DataLoader(dataset_non_sarcastic, batch_size=1, shuffle=True)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    gc.collect()
    torch.cuda.empty_cache()

    model = model.to(device)

    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=-1)

    model = train(4, data_loader_sarcastic, 8, tokenizer, model, device)

    save_model(model, 'sarcastic')

    gc.collect()
    torch.cuda.empty_cache()

    tokenizer_sarcastic = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model_sarcastic = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    tokenizer_non_sarcastic = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model_non_sarcastic = GPT2LMHeadModel.from_pretrained('gpt2-medium')

    model_sarcastic_path = f"sarcastic.pt"
    model_non_sarcastic_path = f"non-sarcastic.pt"

    model_sarcastic.load_state_dict(torch.load(model_sarcastic_path))
    model_non_sarcastic.load_state_dict(torch.load(model_non_sarcastic_path))

    model_sarcastic = model_sarcastic.to(device)
    model_non_sarcastic = model_non_sarcastic.to(device)


    SAR = generate(tokenizer_sarcastic, model_sarcastic, 4000, 'SAR')

    f = open('SAR.txt', 'w')
    for l in SAR:
        f.write(l.replace('SAR', '').replace('<|endoftext|>', '').replace("\n", "").replace(",", " "))
        f.write("\n")
    f.close()

    NON = generate(tokenizer_non_sarcastic, model_non_sarcastic, 4000, 'NON')

    f = open('NON.txt', 'w')
    for l in NON:
        f.write(l.replace('NON', '').replace('<|endoftext|>', '').replace("\n", "").replace(",", " "))
        f.write("\n")
    f.close()


    data = { "tweet": [] , "label": [] }

    for l in NON:
        tweet = l.replace('NON', '').replace('<|endoftext|>', '').replace("\n", "")
        if tweet == "" or tweet == " " or "\n" in tweet:
            pass
        else:
            if tweet[0] == ' ':
                data["tweet"].append(tweet[1:])
            else:
                data["tweet"].append(tweet)
            data["label"].append(0)

    for l in SAR:
        tweet = l.replace('SAR', '').replace('<|endoftext|>', '').replace("\n", "")
        if tweet == "" or tweet == " " or "\n" in tweet:
            pass
        else:
            if tweet[0] == ' ':
                data["tweet"].append(tweet[1:])
            else:
                data["tweet"].append(tweet)
        data["label"].append(1)

    df = pd.DataFrame(data)

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("gpt.csv", index=False)