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
from FineTuner import FineTuner
from Utils import set_seed, get_dataset
from Log import LoggingCallback
from Dataset import SarcasmDataset
import os
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

set_seed(42)

args_dict = dict(
    output_dir="",
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=2,
    eval_batch_size=2,
    num_train_epochs=2,
    gradient_accumulation_steps=4,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)

if __name__ == '__main__':
    train = pd.read_csv('../../Data/Train_Dataset.csv')[["tweet", "sarcastic"]]
    test = pd.read_csv('../../Data/Test_Dataset.csv')

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    ids_nonsarcastic = tokenizer.encode('negative </s>')
    ids_sarcastic = tokenizer.encode('positive </s>')

    train_dataset = SarcasmDataset(tokenizer, train, max_len=512)
    test_dataset = SarcasmDataset(tokenizer, test, max_len=512)
    
    if not os.path.exists('./t5_isarcasm'):
        os.makedirs(path)

    args_dict.update({'output_dir': './t5_isarcasm', 'num_train_epochs':5})
    args = argparse.Namespace(**args_dict)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        precision= 16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)

    trainer.fit(model)

    if not os.path.exists('./t5_base_isarcasm'):
        os.makedirs(path)

    model.model.save_pretrained('t5_base_isarcasm')

    loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    model.model.eval()
    outputs = []
    targets = []
    for batch in tqdm(loader):
        outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                                    attention_mask=batch['source_mask'].cuda(), 
                                    max_length=2)

    dec = [tokenizer.decode(ids) for ids in outs]
    target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
    
    outputs.extend(dec)
    targets.extend(target)

    hasInvalidPrediction = False
    for i, out in enumerate(outputs):
        if out not in ['positive', 'negative']:
            hasInvalidPrediction
            print(f"Detected invalid prediction for sample {i+1} in test.")

    if not hasInvalidPrediction:
        print(metrics.f1_score(targets, outputs, pos_label='positive'))