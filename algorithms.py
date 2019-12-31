import numpy as np
import pandas as pd
import time
import re
from collections import Counter
from linkageSearch import linkageSearch
from createSketchGraph import *
from clusterAlgo import *
from PreprocessStage2 import weightedNodes


def getCountPerCluster(tagsTweet):
    set_per_string=[]
    pat = re.compile(r"#(\w+)")
    set_per_string.append(pat.findall(tagsTweet))
    set_per_string= set_per_string[0]
    set_per_string = [x.lower() for x in set_per_string]
    set_count = Counter(set_per_string)
    for key,val in set_count.items():
        if(val >=1):
            print(key ,"  :      ",val," times in the maximum cluster")




def getClusterSet(test_df,TFIDF,train_df,adjacency_list,cluster_set,density_param):
    
    batch_size=1400
    numberCoreEdges,numberCorePosts,numberClusters,numberEdges = [],[],[],[]
    dbscan_weight={}
    CloudOfCloud,cluster_ids_test,counting_batch = [],[],[]
    sumLengthCorePosts=0
    cluster={}
    print("total batches::",int(test_df.shape[0]/batch_size))
    
    for param in density_param:
        sumLengthCorePosts,sumLengthEdges,sumLengthCoreEdges,clusterCount=0,0,0,0
        id_list={}
        id_set=set()
        for i in range(0, int(test_df.shape[0]/batch_size)):
            start=time.time()
            print("----------------------------------------------------------------------------------------------------------")
            print("Batch: ",i+1)
            
            
            test_batch = test_df.iloc[batch_size*i:batch_size*(i+1)]
            
            test_batch.reset_index(drop=True, inplace=True)
            
            current_time = test_batch.loc[0].Timestamp
            
            linkage_time = time.time()

            print("First, performing linkage search on train data")
            
            final_ls_indices,weight_dict,core_Edges = linkageSearch(train_df, TFIDF, train_df, adjacency_list, param)
            
            print("Now, performing linkage search on test data")
            
            final_ls_indices_testBatch, weight_dict_test,core_Edges_test= linkageSearch(test_batch, TFIDF, train_df, adjacency_list,param)
            
            print("Updating the test ls indices with the train to get one dictionary altogether")
            
            final_ls_indices.update(final_ls_indices_testBatch)
            weight_dict.update(weight_dict_test)
            core_Edges.update(core_Edges_test)
            print("Linkage search takes: ",time.time()-linkage_time)
            
            
            adjacency_list = None
            
            ### Calculating the corePosts, borderPosts
            corePosts_denominator= getCorePostDenominator(train_df, current_time)
            corePosts_denominator_test = getCorePostDenominator(test_batch, current_time)
            corePosts_denominator.update(corePosts_denominator_test)
            
            corePostsPerBatch, borderPostsToCheckPerBatch = getCorePosts(weight_dict,corePosts_denominator)
            
            ### now we have the corePosts list and the coreEdges List, using this we need to know what the corePosts for the sketchGraph are
            
            corePostsSketchGraph = getCorePostsforSketchGraph(corePostsPerBatch,core_Edges)
            borderPostsSketchGraph = getBorderPostsforSketchGraph(final_ls_indices, corePostsSketchGraph, borderPostsToCheckPerBatch)
                        
            ### Now we need to do the cluster part################
            
            ### we have corePosts and borderPosts####
            
            for i in range(test_batch.shape[0]):
                
                key = test_batch.iloc[i].values[0]
                
                if(len(test_batch.loc[test_batch['Id'] == key ].NounList.tolist()[0])==0):
                    continue
                if(key not in final_ls_indices.keys()):
                    continue
                print("Tweet is ",test_batch.loc[test_batch['Id'] == key].text)
                
                corePostsperTestNode = getIntersectionOfTwoList(final_ls_indices[test_batch.loc[i].Id],corePostsSketchGraph)
                borderPostsperTestNode=[]
                
                if(len(corePostsperTestNode)>0):
                    borderPostsperTestNode = getIntersectionOfTwoList(final_ls_indices[test_batch.loc[i].Id] ,borderPostsSketchGraph)
                    print("This post has :", len(corePostsperTestNode)," core posts.")
                    print("This post has :", len(borderPostsperTestNode)," border posts.")
                    corePostsperTestNode = list(set(corePostsperTestNode)) # to remove duplicates if any
                    borderPostsperTestNode = list(set(borderPostsperTestNode))

                    print("Core posts are::")
                    for post in corePostsperTestNode:
                        print(train_df.loc[train_df['Id'] == post].text)

                    cluster = formClusters(cluster, corePostsperTestNode, borderPostsperTestNode)
                else:
                    print("This post does not have any coreposts to form a sketchGraph for itself")
                print("_______________________________________________________________________")

            ### After that need to take care of the deletion of top K posts from train and add the K posts from test to train###
            max_value= 0
            max_cluster_set={}
            cloud=[]
            cid =0
            count=0
            
            for key, value in cluster.items():
                count= count+1
                if max_value < len(value):
                    max_value = len(value)
                    max_cluster_set = value
                    cid = key
            
            print("size of Maximum cluster elements: ", len(max_cluster_set))
            print("Number of clusters we have ",count)
            print("The cluster id that is maximum is ",cid)
            
            print("The cummulative details for all the clusters::")
            
            count=0
            for key,value in cluster.items():
                if(len(value)>0):
                    print("Cluster :", count)
                    print("cluster id is:", key)
                    count=count+1
                    tid=[]
                    for val in value:
                        tid.append(val)
                    tdf = pd.DataFrame(train_df[train_df['Id'].isin(tid)])
                    tstdf = pd.DataFrame(test_df[test_df['Id'].isin(tid)])
                    tdf.append(tstdf)
                    tdf.dropna()
                    getCountPerCluster(tdf.text.to_string())
                    print("______________________________________________________________________________________")
            
            
            comment_words = ""
            for column in tdf[['NounList']]:
                for i in tdf[column]:
                    x = " ".join(map(str, i))
                    comment_words = comment_words + x + " "
            cloud.append(comment_words)
            CloudOfCloud.append(cloud)
            
            train_df = train_df.append(test_batch)
            idToDeleteFromClusterSet = train_df.Id[:batch_size].values.tolist()
            train_df.reset_index(inplace=True, drop = True)
            train_df, adjacency_list,TFIDF = deletePost(train_df,batch_size)
            cluster = deleteClusterSet(idToDeleteFromClusterSet,cluster)
                  
            print("Time taken for this batch: ", time.time()-start)
    return cluster, CloudOfCloud
        