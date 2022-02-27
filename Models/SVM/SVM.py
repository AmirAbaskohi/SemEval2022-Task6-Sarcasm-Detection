import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import mxnet as mx
from bert_embedding import BertEmbedding
from nltk.tokenize import word_tokenize
import gensim
import itertools

def mean(z):
    return sum(itertools.chain(z))/len(z)

def get_result(y_test, y_pred):
    print(" accuracy is: ", accuracy_score(y_test, y_pred))
    print(" f1-score is: ", f1_score(y_test, y_pred))

def embeddToBERT(text):
    sentences = re.split('!|\?|\.',text)
    sentences = list(filter(None, sentences)) 

    result = bert(sentences, 'avg') # avg is refer to handle OOV

    bert_vocabs_of_sentence = []
    for sentence in range(len(result)):
        for word in range(len(result[sentence][1])):
            bert_vocabs_of_sentence.append(result[sentence][1][word])
    feature = [mean(x) for x in zip(*bert_vocabs_of_sentence)]
  
    return feature

def embeddToWord2Vec(text):
    words = word_tokenize(text)
    
    result = [w2v_with_stop_model.wv[w] for w in words if w in w2v_with_stop_model.wv.vocab]
    
    feature = [mean(x) for x in zip(*result)]
    return feature

def wordTokenize(text):
  return word_tokenize(text)

if __name__ == '__main__':
    class_weight= {1: 3, 0: 1}
    
    train = pd.read_csv('../../Data/Train_Dataset.csv')
    test = pd.read_csv('../../Data/Test_Dataset.csv')
     
    X_train, y_train = train["tweet"], train["sarcastic"]
    X_test, y_test = test["tweet"], test["sarcastic"]
    
    
    concat_dataset = pd.concat([train, test]).reset_index(drop=True)
    
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(concat_dataset["tweet"].values.astype('U'))
    
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count_vect.get_feature_names_out(),columns=["idf_weights"]) 
    df_idf.sort_values(by=['idf_weights'])

    # result with count vectorizer
    XX_train = X_counts.toarray()[:len(train)]
    XX_test = X_counts.toarray()[len(train):]

    clf = SVC(C=10, kernel='rbf', class_weight=class_weight)
    clf.fit(XX_train, y_train)
    y_pred = clf.predict(XX_test)
    get_result(y_test, y_pred)
    
    

    # result with TF-IDF
    XX_train = X_tfidf.toarray()[:len(train)]
    XX_test = X_tfidf.toarray()[len(train):]

    clf = SVC(C=10, kernel='rbf', class_weight=class_weight)
    clf.fit(XX_train, y_train)
    y_pred = clf.predict(XX_test)
    get_result(y_test, y_pred)



    # result with BERT(Word Tokenization format)
    ctx = mx.gpu(0)
    bert = BertEmbedding(ctx=ctx)

    bert_word_training_features_train = X_train.apply(embeddToBERT)
    feature_train = [x for x in bert_word_training_features_train.transpose()]
    
    bert_word_training_features_test = X_test.apply(embeddToBERT)
    feature_test = [x for x in bert_word_training_features_test.transpose()]

    XX_train = np.asarray(feature_train)
    XX_test = np.asarray(feature_test)
    
    clf = SVC(C=10, kernel='rbf', class_weight=class_weight)
    clf.fit(XX_train, y_train)
    y_pred = clf.predict(XX_test)
    get_result(y_test, y_pred)



    # result with Word2Vec

    #train
    words_train = X_train.apply(wordTokenize)
    w2v_with_stop_model= gensim.models.Word2Vec(words_train, min_count = 2, size = 100, window = 5) 

    word2vec_with_stop_training_features_train = X_train.apply(embeddToWord2Vec)

    feature_train = []
    deleted_indexes_train = []
    i = 0
    for x in word2vec_with_stop_training_features_train.transpose():
        if x != []:
            feature_train.append(x)
        else:
            deleted_indexes_train.append(i)
        i += 1
    XX_train = np.asarray(feature_train)


    #test
    words_test = X_test.apply(wordTokenize)
    w2v_with_stop_model = gensim.models.Word2Vec(words_test, min_count = 2, size = 100, window = 5) 

    word2vec_with_stop_training_features_test = X_test.apply(embeddToWord2Vec)

    feature_test = []
    deleted_indexes_test = []
    i = 0
    for x in word2vec_with_stop_training_features_test.transpose():
        if x != []:
            feature_test.append(x)
        else:
            deleted_indexes_test.append(i)
        i += 1
    XX_test = np.asarray(feature_test)
    
    
    clf = SVC(C=10, kernel='rbf', class_weight=class_weight)
    clf.fit(XX_train, y_train.drop(deleted_indexes_train))
    y_pred = clf.predict(XX_test)
    get_result(y_test.drop(deleted_indexes_test), y_pred)