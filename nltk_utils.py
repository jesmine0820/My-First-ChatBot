import nltk
#nltk.download('punkt_tab')
import numpy as np

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

#Split sentence into words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#Find the base word
def stem(word):
    return stemmer.stem(word.lower())

#Check whether the word exists
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag
