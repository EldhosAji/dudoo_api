import nltk
from nltk.stem.porter import PorterStemmer 
stemmer = PorterStemmer()
import numpy as np

def TOCKENIZE(sentence):
    return nltk.word_tokenize(sentence)

def STEM(word):
    return stemmer.stem(word.lower())

def WORD_CONTAINER(tokenized_sentence,WORD_LIST):
    tokenized_sentence_words = [STEM(w) for w in tokenized_sentence]
    wordlist = np.zeros(len(WORD_LIST),dtype = np.float32)
    for idx,w in enumerate(WORD_LIST):
        if w in tokenized_sentence_words:
            wordlist[idx] = 1
    return wordlist