from random import sample, random, shuffle
import json

class TextMutant():
    def __init__(self):
        synm_filepath = 'synm.json'

        with open(synm_filepath) as json_file:
            self.synoynms = json.load(json_file)
    
    def remove_words(self, sentence, mode = "k-random", k = 0.1, prob = 0.1):
        word_list = sentence.split()
        new_words = []
        if mode == "prob":
            for i in word_list:
                if (random() < prob):
                    continue
                new_words.append(i)
        if mode == "k-random":
            num = round(len(word_list) * k)
            random_index = sample(list(range(len(word_list))), num)
            new_words = list( word_list[i] for i in range(len(word_list)) if i not in random_index )
        return " ".join(new_words)

    def shuffle_words(self, sentence, prob = 0.1):
        word_list = sentence.split()
        indexes = list(range(len(word_list)))
        if (random() < prob):
            shuffle(indexes)
        new_words = list( word_list[i] for i in indexes )
        return " ".join(new_words)
    
    def replace_words(self, sentence, mode = "k-random", k = 0.1, prob = 0.1):
        word_list = sentence.split()
        new_words = []
        if mode == "prob":
            for i in word_list:
                self.synoynms.setdefault(i, [])
                if (random() < prob and len(self.synoynms[i]) > 0):
                    new_words.append(sample(self.synoynms[i], 1)[0])
                    continue
                new_words.append(i)
        if mode == "k-random":
            num = round(len(word_list) * k)
            indexes = list(range(len(word_list)))
            shuffle(indexes)
            new_words = word_list[:]
            for i in indexes:
                self.synoynms.setdefault(word_list[i], [])
                if (num > 0 and len(self.synoynms[word_list[i]]) > 0):
                    new_words[i] = sample(self.synoynms[word_list[i]], 1)[0]
                    num -= 1
        return " ".join(new_words)

    def create_new_sentence(self, sentence, flags,  shuffle_prob = 1, replace_k = 0.5, remove_k = 0.3):
      if flags[0] == '1':
        sentence = self.remove_words(sentence, k = remove_k)
      if flags[1] == '1':
        sentence = self.replace_words(sentence, k = replace_k)
      if flags[2] == '1':
        sentence = self.shuffle_words(sentence, prob=shuffle_prob)
      return sentence
    
    def create_new_dataset(self, dataset, flags):
      dataset_copy = dataset.copy()
      for i in range(len(dataset['tweet'])):
        dataset_copy['tweet'].iloc[i] = self.create_new_sentence(dataset_copy['tweet'].iloc[i], flags)
      return dataset_copy