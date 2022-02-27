import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def cleaner(X, flag):
  X_copy = X.copy()
  
  porter_stemmer = PorterStemmer()
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))
  pancs1 = [',', '.', ';', ':', '!', '?', '"', '-', '“', '”', '/', '\\', '(', ')', '_', '+', '*', '^', '¯', 'ツ', '…', '&',
           '[', ']', '{', '}', '~', '—', '–', '|', '‼', '≠', '°', '•', '=', 'ï', '∞', 'ú', 'ð', '​'] 
  
  pancs = pancs1

  for i in range(len(X_copy['tweet'])):
    X_copy['tweet'].iloc[i] = X_copy['tweet'].iloc[i].lower()

    # flag[6] : remove_tags
    if(flag[6] == '1'):
      without_tag = []
      for j in X_copy['tweet'].iloc[i].split():
        if '@'not in j:
          without_tag.append(j)
      X_copy['tweet'].iloc[i] = " ".join(without_tag)

    # flag[5] : remove_hashtags
    if(flag[5] == '1'):
      without_hashtags = []
      for j in X_copy['tweet'].iloc[i].split():
        if '#'not in j:
          without_hashtags.append(j)
      X_copy['tweet'].iloc[i] = " ".join(without_hashtags)

    # flag[4] : remove_links
    if(flag[4] == '1'):
      without_links = []
      for j in X_copy['tweet'].iloc[i].split():
        if 'https'not in j:
          without_links.append(j)
      X_copy['tweet'].iloc[i] = " ".join(without_links)

    # flag[3] : remove_punctuations
    if(flag[3] == '1'):
        temp = []
        for j in list(X_copy['tweet'].iloc[i]):
            if j in ["‘", "’", "'", "’"]:
                temp.append('')
            elif j in pancs:
                temp.append(' ')
            else:
                temp.append(j)
        X_copy['tweet'].iloc[i] = "".join(temp)

    # flag[2] : stemming
    if(flag[2] == '1'):
      after_stemming = []
      for j in X_copy['tweet'].iloc[i].split():
          after_stemming.append(porter_stemmer.stem(j))
      X_copy['tweet'].iloc[i] = " ".join(after_stemming)

    # flag[1] : lemmatizing
    if(flag[1] == '1'):
      wordsList = X_copy['tweet'].iloc[i].split()
      wordsList_pos = nltk.pos_tag(wordsList)

      after_lemmatizing = []
      for word, pos in wordsList_pos:
        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos == '':
            lemm = lemmatizer.lemmatize(word)
        else:
            lemm = lemmatizer.lemmatize(word, wordnet_pos)

        after_lemmatizing.append(lemm)
      X_copy['tweet'].iloc[i] = " ".join(after_lemmatizing)

    # flag[0] : remove_stopwords
    if(flag[0] == '1'):
      without_stopwords = []
      for j in X_copy['tweet'].iloc[i].split():
        if j.lower() not in stop_words:
          without_stopwords.append(j)
      X_copy['tweet'].iloc[i] = " ".join(without_stopwords)

  return X_copy

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

if __name__ == "__main__":
    dataset_path = '../Data/Main Dataset/Train_Dataset.csv'
    df = pd.read_csv(dataset_path)[["tweet", "sarcastic"]]
    df = df[df['tweet'].notna()]
    X, y = df[["tweet"]], df[["sarcastic"]]

    for i in range(128):
        flag = "".join(['0']*(7 - len(bin(i)[2:]))) + bin(i)[2:]
        cleaner(X, flag).to_csv("MutantData-" + flag + ".csv")