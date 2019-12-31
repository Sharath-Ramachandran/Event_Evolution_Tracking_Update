import numpy as np
import ast

"""
This function extracts  k nouns with lowest minimum TF-IDF score.
"""

def getMinTFIDFOfNounList(nounList,TFIDF,k=10):
    min_K = []
    count=0
    for i in nounList:
        tf = TFIDF.get(i.lower())
        if tf!=None:
            min_K.append(tf)
    
    if min_K == []:
        return -1
    else:
        min_K = np.asarray(min_K)
        index_words= np.argsort(min_K)

        if len(min_K)<k:
            k = len(min_K)
        list_min_k = []
        for i in range(len(index_words)):
            if(nounList[index_words[i]] not in list_min_k):
                list_min_k.append(nounList[index_words[i]])
                count =count+1
                if(count== k):
                    break
        return list_min_k

    
"""
Adding a coloumn "NounWords" containing list of nouns for each post  in our dataframe.
"""

def addNounWordInDataFrame(fd):
    fd['NounWords']=''
    for i in range(len(fd)):
        fd['NounWords'][i]= ' '.join(fd.NounList[i])
    return fd


"""
Function for 
"""
def getClosestRowfromDF(fd, weightedNouns):
    
    pattern = '|'.join(['(?i)' + word for word in weightedNouns])
    indices = fd.NounWords.str.contains(pattern)
    indices = indices.values
    indices =np.argwhere(indices==True)
    
    if len(indices) == 0:
        ser = (fd.NounWords.str.contains(pattern))
        ser= ser.values
        indices =np.argwhere(ser==True)
    return indices



"""

"""
def getNearbyneighbour(indices,fd,adjacency_list):
    nDash = []
    tweetDash=[]
    timeStamp=[]
    nDashSize = 100
    
    for index in range(0,len(indices)):
        tweet_Id = fd.iloc[indices[index][0]].Id
#         print("Tweet text is:::",fd.iloc[indices[index][0]].text)
        nDash.append(tweet_Id)
        nDash=list(set(nDash))
        if(adjacency_list.get(tweet_Id)== None):
            continue
#         print("Tweet text is:::",fd.iloc[indices[index][0]].text)
#         print("they both are similar!")
#         print()
#         print()
        
#         print("Nounlist of this one is :::",fd.iloc[indices[index][0]].NounList)
#         print("Number of adjacency list neighbours::::", len(set(adjacency_list.get(tweet_Id))))
        nDash.extend(set(adjacency_list.get(tweet_Id)))
        nDash=list(set(nDash)) 
        
        if len(nDash) < nDashSize:
            continue
        else:
            nDash = nDash[:nDashSize]
            break
#     print("NDash is ::", nDash)
    for i in nDash:
        tweet_text=fd.loc[fd['Id'] == i, 'NounList'].tolist()[0]
        tweetDash.append(tweet_text)
        
        timeStamp_tweet=fd.loc[fd['Id'] == i, 'Timestamp'].tolist()[0]
        timeStamp.append(timeStamp_tweet)
 
    return nDash, tweetDash, timeStamp



"""
function to calculate Linkage score between the post.
getLinkageScore(listOfNoun,test_dataframe.iloc[i].Timestamp,tweetDash, timeStamp)
"""

def getLinkageScore(listOfNoun_testNode,testNode_time,listOfNoun_neighbour,neighbour_time):
    testNode_time=int(testNode_time) # testNode_time = int(testNode_time)
    neighbour_time=int(neighbour_time) # neighbour_time = int(neighbour_time)
    
    intersection = len(list(set(listOfNoun_testNode).intersection(listOfNoun_neighbour)))
    union = (len(listOfNoun_testNode) + len(listOfNoun_neighbour)) - intersection

    time_diff = np.abs(testNode_time-neighbour_time)
    D=1+time_diff
    
    denm=(union)*D
    
    ls_val=intersection/denm
    
    
    return ls_val


"""
Function to find set of neighbours for each post based on linkage search.
Returning a dictionary with key as tweet ids and values as neighbouring post's tweet id.

"""


def linkageSearch(test_dataframe,TFIDF,fd,adjacency_list,e1):
    final_ls_indices={}
    core_Edges={}
    
    e0= 0.0000002
    
    weight_dict={}
    for i in range(test_dataframe.shape[0]):
#         print("______________________________________________________________________________________________________")
        summation=0
        count=0
        k = 10
        nDashSize = 100
        coreEdgesList=[]
        final_indices=[]
        e0_scores=[]
        nDash = []
        tweetDash=[]
        timeStamp=[]
#         print("Tweet Text we came to see  is",test_dataframe.iloc[[i]].text)
        listOfNoun = test_dataframe.iloc[[i]].NounList.tolist()[0]
#         print("Noun list is:::", listOfNoun)
        weightedNouns = getMinTFIDFOfNounList(listOfNoun,TFIDF)
#         print("Nouns we consider are :::", weightedNouns)
        if weightedNouns == -1:
            continue
        
        indices= getClosestRowfromDF(fd,weightedNouns)
#         if len(indices)==0:
#             print("Outliers")
        if len(indices)>0:
            nDash, tweetDash, timeStamp = getNearbyneighbour(indices,fd,adjacency_list)
        
        for m,j, k in zip(nDash,tweetDash, timeStamp):
            ls_val=getLinkageScore(listOfNoun,test_dataframe.iloc[i].Timestamp,j,k)
#             print("LS_Score ::::",ls_val)
            if (ls_val>=e0):
                final_indices.append(m)
                e0_scores.append(ls_val)
            if(ls_val>=e1): # density parameter is passed
#                 print("Came for core edges code::::")
                coreEdgesList.append(m)
                
                
        ''' sort the scores based on descending values and then sort the id's based on that sorting'''
        e0_scores = np.asarray(e0_scores)
        final_indices = np.asarray(final_indices)
        arr1inds = e0_scores.argsort()
        e0_scores = e0_scores[arr1inds[::-1]]
        final_indices = final_indices[arr1inds[::-1]]
        e0_scores = e0_scores.tolist()
        e0_scores = e0_scores[:20]
#         print("Getting the summation to calculate the weight of a post::")
        summation = sum(e0_scores)
        final_indices = final_indices.tolist()
        final_indices = final_indices[:20] # get the top 20 indices in Linkage search
#         print("Number of linkage search neighbours obtained for  this post  is ",len(final_indices))
#         print("___________________________________________________________________________________________________")
        
        core_Edges.update({test_dataframe.iloc[i].Id: coreEdgesList})
        weight_dict.update({test_dataframe.iloc[i].Id:summation})
        final_ls_indices.update({test_dataframe.iloc[i].Id:final_indices})
    return final_ls_indices, weight_dict,core_Edges