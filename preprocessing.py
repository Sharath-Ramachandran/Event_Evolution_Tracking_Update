import nltk
import enchant
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import emoji
from nltk.stem import PorterStemmer
import datetime
import sys
from nltk.tokenize import word_tokenize, sent_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

"""
This functions takes the tweets as a input and returns list of nouns.
"""

def getNounsList(tweet):
    nouns=[]
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(tweet)
    for i in tokenized:  
        wordsList = nltk.word_tokenize(i)
        wordsList = [w for w in wordsList if not w in stop_words and w.isalpha()]  
        newList=[]
        for w in wordsList:
            newList.append(w)
        for word,pos in nltk.pos_tag(newList):
            if (pos == 'NN' or pos == 'NNP'):
                nouns.append(word)
            if(pos=='NNS' or pos=='NNPS'):
                nouns.append(ps.stem(word))
    return nouns


"""
This functions helps to get the nouns out of tweets.
"""

def getNounsfromTweets(tweet):
    if(type(tweet)==str):
        tweet = tweet.lower()
        tweet= re.sub(r'http\S+', '',tweet)
        tweet=re.sub(r'pic.\S+','',tweet)
        NounsList= getNounsList(tweet)
    else:
        NounsList=[]
    return NounsList