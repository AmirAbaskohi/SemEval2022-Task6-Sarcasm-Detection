import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.svm import SVC
import mxnet as mx
from bert_embedding import BertEmbedding
from nltk.tokenize import word_tokenize
import gensim
import itertools


def report_acc_cv(clf, X, y, model_name, cv=10):    
    metrics = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall'])
    
    fit_time = metrics['fit_time']
    score_time = metrics['score_time']
    acc = metrics['test_accuracy']
    accb = metrics['test_balanced_accuracy']
    f1 = metrics['test_f1']
    p = metrics['test_precision']
    r = metrics['test_recall']
    
    print("Metrics for " + model_name)
    print(" fit_time is: %.2f s +- %.2f s" %(np.mean(fit_time),np.std(fit_time)))
    print(" score_time is: %.2f s +- %.2f s\n" %(np.mean(score_time),np.std(score_time)))
    print(" accuracy is: %.2f%% +- %.2f%%" %(np.mean(acc)*100,np.std(acc)*100))
    print(" balanced accuracy is: %.2f%% +- %.2f%%" %(np.mean(accb)*100,np.std(accb)*100))
    print(" f1-score is: %.2f%% +- %.2f%%" %(np.mean(f1)*100,np.std(f1)*100))
    print(" precision is: %.2f%% +- %.2f%%" %(np.mean(p)*100,np.std(p)*100))
    print(" recall is: %.2f%% +- %.2f%%" %(np.mean(r)*100,np.std(r)*100))

def embeddToBERT(text):
    sentences = re.split('!|\?|\.',text)
    sentences = list(filter(None, sentences)) 

    if bert_version == 'WORD':
        result = bert(sentences, 'avg') # avg is refer to handle OOV
    
        bert_vocabs_of_sentence = []
        for sentence in range(len(result)):
            for word in range(len(result[sentence][1])):
                bert_vocabs_of_sentence.append(result[sentence][1][word])
        feature = [mean(x) for x in zip(*bert_vocabs_of_sentence)]

    elif bert_version == 'SENTENCE':
        result = bert_transformers.encode(sentences)
        feature = [mean(x) for x in zip(*result)]
  
    return feature

def mean(z):
    return sum(itertools.chain(z))/len(z)

def embeddToWord2Vec(text):
    words = word_tokenize(text)
    
    if embedding is 'WORD2VEC_WITH_STOP':
        result = [w2v_with_stop_model.wv[w] for w in words if w in w2v_with_stop_model.wv.vocab]
    else:
        result = [w2v_no_stop_model.wv[w] for w in words if w in w2v_no_stop_model.wv.vocab]
    
    feature = [mean(x) for x in zip(*result)]
    return feature

def wordTokenize(text):
  return word_tokenize(text)

if __name__ == '__main__':

    # dataset address
    dataset_path = '../Data/Train_Dataset.csv'
    df = pd.read_csv(dataset_path)[["tweet", "sarcastic"]]
    df = df[df['tweet'].notna()]
    X, y = df[["tweet"]], df[["sarcastic"]]

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X['tweet'].values.astype('U'))
    X_train_counts.toarray()

    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count_vect.get_feature_names(),columns=["idf_weights"]) 
    df_idf.sort_values(by=['idf_weights'])


    # result with count vectorizer
    X_train = X_train_counts
    class_weight= {1: 3, 0: 1}

    clf = SVC(C=10, kernel='rbf', class_weight=class_weight)
    report_acc_cv(clf, X_train, y.values.ravel(), "svm-count-vectorizer")

    # result with TF-IDF
    X_train = X_train_tfidf
    class_weight= {1: 3, 0: 1}

    clf = SVC(C=10, kernel='rbf', class_weight=class_weight)
    report_acc_cv(clf, X_train, y.values.ravel(), "svm-TF-IDF")

    # result with BERT(Word Tokenization format)
    bert_version = 'WORD'

    ctx = mx.gpu(0)
    bert = BertEmbedding(ctx=ctx)

    bert_word_training_features = X['tweet'].apply(embeddToBERT)

    feature = [x for x in bert_word_training_features.transpose()]
    bert_word_training_features = np.asarray(feature)

    class_weight= {1: 3, 0: 1}

    clf = SVC(C=10, kernel='rbf', class_weight=class_weight)
    report_acc_cv(clf, bert_word_training_features, y.values.ravel(), "svm-BERT")

    # result with Word2Vec

    embedding = 'WORD2VEC_WITH_STOP'
    words = X['tweet'].apply(wordTokenize)
    w2v_with_stop_model = gensim.models.Word2Vec(words, min_count = 2, size = 100, window = 5)

    word2vec_with_stop_training_features = X['tweet'].apply(embeddToWord2Vec)

    feature = []
    deleted_indexes = []
    i = 0
    for x in word2vec_with_stop_training_features.transpose():
        if x != []:
            feature.append(x)
        else:
            deleted_indexes.append(i)
        i += 1
    word2vec_with_stop_training_features = np.asarray(feature)

    class_weight= {1: 3, 0: 1}

clf = SVC(C=10, kernel='rbf', class_weight=class_weight)
report_acc_cv(clf, word2vec_with_stop_training_features, y.drop(deleted_indexes).values.ravel(), "svm-Word2Vec")