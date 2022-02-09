import pandas as pd
from keras.layers import LSTM, Activation, Dense, Embedding
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from random import shuffle
from sklearn.model_selection import KFold

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


if __name__ == '__main__':
    
    #dataset address
    dataset_path = '../Data/Train_Dataset.csv'
    dataset = pd.read_csv(dataset_path)[["tweet", "sarcastic"]]

    dataset = dataset.dropna(axis = 0)
    dataset.reset_index(drop=True, inplace=True)

    X_data = dataset.tweet
    Y_data = dataset.sarcastic

    vocab_size = 10000
    embedding_dim = 16
    max_length = 150
    trunc_type = 'post'


    tokenizer = Tokenizer(num_words = vocab_size)
    tokenizer.fit_on_texts(X_data)
    sequences = tokenizer.texts_to_sequences(X_data)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

    X = padded
    Y = Y_data

    model_lstm = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    kfold = KFold(n_splits=10, shuffle=True)

    fold_no = 1
    for train, test in kfold.split(X):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        class_weights = {1:3, 0:1}
        train = train.tolist()
        test = test.tolist()
        shuffle(test)
        shuffle(train)

        model_lstm = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model_lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', f1_m])
        
        history = model_lstm.fit(X[train], Y[train], batch_size=32, epochs=5, validation_data=(X[test], Y[test]), class_weight=class_weights,shuffle=True)
        
        fold_no = fold_no + 1
    
    print(history.history)