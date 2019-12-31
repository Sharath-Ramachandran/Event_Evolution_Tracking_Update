'''
Need to write about sketchGraph
---------------------------------
1. This needs to be done for the entire data that we posses. 
2. now we have ls indices for both train and test data. 
3. We also need to find the weight of all the posts that we are currently using the formula given in the paper.
4. There now, we have to update t as the time of the current test batch first post timestamp.
5. Then we use a threshold to say if a post is core or border or noise
6. If core, add it to clusters. If border, then need to see if any of its posts from ls indices are core, then it will be also be added to the same post.
7. Also, we need to delete the posts that we previously core posts but now became noise because of time.( Need to set the threshold appropraitely).
8. Then, we print all the clusters that have size(number of posts) greater than a threshold(10) in them. 

'''


import numpy as np
import pandas as pd
import ast
import time



def getBorderPostsforSketchGraph(final_ls_indices,corePosts,checkForBorderPosts):
    borderPosts=[]
    for key in checkForBorderPosts:
                    if(len(getIntersectionOfTwoList(final_ls_indices[key], corePosts))>0):
                        borderPosts.append(key)
    return borderPosts



def getCoreEdgesListFromDictionary(coreEdges):
    keys_list=[]
    for k in coreEdges.keys():
        keys_list.append(k)
    vals_list = coreEdges.values()
    vals_list = [inner for outer in vals_list for inner in outer]
    l1 = np.asarray(vals_list)
    l2 = np.asarray(keys_list)
    return(np.union1d(np.unique(l1),np.unique(l2))).tolist()



def getIntersectionOfTwoList(list1,list2):
    l1 = np.asarray(list1)
    l2 = np.asarray(list2)
    intersection_list = (np.intersect1d(np.unique(l1),np.unique(l2))).tolist()
    return intersection_list

def getCorePostDenominator(df, currentTime):
    denominator={}
    for i in range(df.shape[0]):
        denominator[df.loc[i].Id]=np.abs(df.loc[i].Timestamp-currentTime)+1
    return denominator

def getWeight(summationValue, denominator):
    return summationValue/denominator
    
    
def getCorePostsforSketchGraph(corePostsList, coreEdges):
    ## here coreEdges is a dictionary
    listOfCoreEdgesPosts = getCoreEdgesListFromDictionary(coreEdges)
    listOfCorePostsSketchGraph = getIntersectionOfTwoList(listOfCoreEdgesPosts, corePostsList)
    return listOfCorePostsSketchGraph

def getCorePosts(weight_dict,denominator):
    parameter_corePosts=0.0000002 # the parameter written in paper
    corePosts=[]
    checkForBorderPosts=[]
    
    for key, value in weight_dict.items():
        if(getWeight(value,denominator[key])>=parameter_corePosts):
            corePosts.append(key)
        else:
            checkForBorderPosts.append(key)
    
    return corePosts,checkForBorderPosts

        
        