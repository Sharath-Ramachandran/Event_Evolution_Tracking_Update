import pandas as pd
import numpy as np
from preprocessing import getNounsfromTweets


"""
This function is removing duplicate entries with same username and timestamp.
"""
def RemovingDuplicatesandCreateDataFame(tweets):
    tweet=[]
    for i in range(0,tweets.shape[0]):
        tweet.append(getNounsfromTweets(tweets[i]))
    return tweet

def weightedNodes(similarity_matrix,tweet_Id):#here weighted means tweetId
    return similarity_matrix*tweet_Id