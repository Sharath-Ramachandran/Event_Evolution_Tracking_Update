import numpy as np
import pandas as pd
from createSketchGraph import getIntersectionOfTwoList
from graphProcessing_Recifading import fadingSimilarity ,getNodeFromWeightedSimilarityMatrix , getTFIDF
from PreprocessStage2 import weightedNodes



def deleteClusterSet(idToDeleteFromClusterSet, cluster_set):
    for key,val in cluster_set.items():
        for ids in idToDeleteFromClusterSet:
            if ids in val:
                val.remove(ids)
    return cluster_set
                          
                
                

def deletePost(train_df,batch_size=100):
    train_df1 = train_df.iloc[batch_size:]
    train_df1.reset_index(inplace=True, drop = True) # mandatory to do as it causes problem
    timestamp = train_df1.Timestamp.values
    
    similarity_matrix = fadingSimilarity(train_df1.NounList.values,timestamp,train_df1.shape[0])
    
    tweetId = train_df1.Id.values.astype(np.int64)
    
    weightedSimilarityMatrix = weightedNodes(similarity_matrix,tweetId)
    adjacency_list = getNodeFromWeightedSimilarityMatrix(weightedSimilarityMatrix ,tweetId)
    similarity_matrix=None
    weightedSimilarityMatrix=None
    TFIDF = getTFIDF(train_df1.NounList)
    return train_df1,adjacency_list,TFIDF
    
    
                
def formClusters(cluster, corePostsperTestNode, borderPostsperTestNode):
    if(len(corePostsperTestNode)==0):
        return cluster
    else:
        flag=0
        cluster_id =1
        cluster_to_change={}
        if(bool(cluster)==False):
            cluster[1]=set(corePostsperTestNode)
            if(len(borderPostsperTestNode)>0):
                cluster[1].update(set(borderPostsperTestNode))
        else:
            for key, value in cluster.items():
                cluster_id=cluster_id+1
                intersectionVal = getIntersectionOfTwoList(list(value),corePostsperTestNode)
                if len(intersectionVal)>0:
                    flag=1
                    cluster_to_change[key]=value
            if(flag==0):
                if(len(corePostsperTestNode)>0):
                    cluster[cluster_id] =set(corePostsperTestNode)
                    if(len(borderPostsperTestNode)>0):
                        cluster[cluster_id].update(set(borderPostsperTestNode))
            else:
                cluster = changetheClusters(cluster_to_change, cluster,corePostsperTestNode,borderPostsperTestNode)
    return cluster



def changetheClusters(cluster_to_change, cluster_dict,core_set,border_set):
        
    ## remove the key, value of cluster_to_change from cluster_dict
    for key in cluster_to_change:
        del cluster_dict[key]
    ## merge all the values in cluster_to_change to a single value with key as first key of cluster_to_change
    cluster_merge=set()
    for key in cluster_to_change:
        cluster_merge.update(cluster_to_change[key])
    
    cluster_merge.update(set(core_set))
    cluster_merge.update(set(border_set))
    
    cluster_dict[list(cluster_to_change.keys())[0]] = cluster_merge #put all the merged values in the first key of cluster_to_change
    cluster_dict = orderKeysDictionary(cluster_dict)
    return cluster_dict
            
def orderKeysDictionary(cluster_dict):
    dummy_set={}
    i=1
    for key in cluster_dict:
        if(len(cluster_dict[key])>0):
            dummy_set[i] = cluster_dict[key]
            i=i+1
    cluster_dict = dummy_set
    del dummy_set
    return cluster_dict