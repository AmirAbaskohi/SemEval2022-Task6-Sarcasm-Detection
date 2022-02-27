import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import random
import pandas as pd
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dataset_embedding(dataset_path, tokenizer, batch_size=32):
    dataset = pd.read_csv(dataset_path)[["tweet", "sarcastic"]]
    dataset = dataset[dataset['tweet'].notna()]
    
    tokenized_tweets = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweet)) for tweet in dataset['tweet']]
    
    tweets_with_len = [[tweet, dataset['sarcastic'].iloc[i], len(tweet)] for i, tweet in enumerate(tokenized_tweets)]
    random.Random(42).shuffle(tweets_with_len)
    
    tweets_with_len.sort(key=lambda x: x[2])
    sorted_tweets_labels = [(tweet_lab[0], tweet_lab[1]) for tweet_lab in tweets_with_len] # remove tweet len
    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_tweets_labels, output_types=(tf.int32, tf.int32))
    
    return processed_dataset.padded_batch(batch_size, padded_shapes=((None, ), ()))

def prepare_datasets(train_path, test_path):
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    
    dataset_train = dataset_embedding(train_path, tokenizer)
    dataset_test = dataset_embedding(test_path, tokenizer)
    
    return dataset_train, dataset_test, tokenizer