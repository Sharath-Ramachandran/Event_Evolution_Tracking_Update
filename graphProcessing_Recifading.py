import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer



def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union == 0:
        return 0.0
    return float(intersection / union)







def fadingSimilarity(NounList,TimeStamp,total_len):
    e0 = 0.002
    similarity_matrix = np.zeros((total_len,total_len))
    for i in range(total_len):
        for j in range(i+1,total_len):
            time_diff = np.abs(TimeStamp[i]-TimeStamp[j])
#             time_diff= time_diff/(3600*24*60)
            similarity_matrix[i,j] = jaccard_similarity(NounList[i],NounList[j])/((np.abs(time_diff)+1))#np.exp(np.abs(time_diff))
#             print("For ",i," and ",j," similarity is",similarity_matrix[i,j])
    np.fill_diagonal(similarity_matrix,0)
    similarity_matrix[similarity_matrix<e0] = 0.0
    similarity_matrix[similarity_matrix>=e0]=1.0
    
    
    return (similarity_matrix+similarity_matrix.T).astype(np.int64)






def getNodeFromWeightedSimilarityMatrix(res,tweetId):
    graph = defaultdict(list)
    edges = set()

    for i, v in enumerate(res, 1):
        for j, u in enumerate(v, 1):
            if u != 0 and frozenset([i, j]) not in edges:
                edges.add(frozenset([i, j]))
                graph[tweetId[i-1]].append(u)
    s = dict(np.ndenumerate(graph))
    x = s.values()
    final_dict = list(x)
    return final_dict[0]


def joinStrings(stringList):
    return ' '.join(string for string in stringList)






def getTFIDF(values_in_list):
    noun_single_list = []
    for i in range(len(values_in_list)):
        noun_single_list.append(joinStrings(values_in_list[i]))
    tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    txt_fitted = tf.fit_transform(noun_single_list)
    idf = tf.idf_
    return dict(zip(tf.get_feature_names(), idf))



