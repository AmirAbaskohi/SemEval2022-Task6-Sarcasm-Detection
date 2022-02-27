import Utils

import tensorflow as tf

train_path = '../Data/Train_Dataset.csv'
test_path = '../Data/Test_Dataset.csv'

train_data, test_data, tokenizer = Utils.dataset_embedding(train_path, test_path)

Blstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.vocab), 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

Blstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', Utils.f1_m])

print(Blstm.summary())

Blstm.fit(train_data, epochs=10, validation_data=test_data, class_weight={1:4, 0:1})
