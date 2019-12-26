import random
import numpy as np
import pandas as pd
import math

from scipy.spatial.distance import pdist
from sklearn.metrics import confusion_matrix, adjusted_rand_score, v_measure_score, fowlkes_mallows_score, f1_score, \
    mutual_info_score

def getdata(file_path):
    print(file_path)
    data = pd.read_csv(file_path, sep='\t', header=None,encoding = "utf-8")
    labels = data.iloc[:, -1]
    data_matrix = data.iloc[:, :-2]
    return data_matrix, labels

def RAND(low,up):
    return random.uniform(low, up)

def bound(low,up,val):
    if val<low:
        val=low
    elif val>up:
        val=up
    return val

def max_index(list,start,end):
    max_val=list[start]
    index=start
    for i in range(start+1,end):
        if list[i]>max_val:
            max_val=list[i]
            index=i
    return index

#相似度计算
def Cosin(a,b):
    # return 1-pdist(np.vstack([a,b]),'cosine')
    v1=np.sum(a*b)
    v2=np.linalg.norm(a)*np.linalg.norm(b)
    # 分子分母接近与0则返回0
    # return v1 / v2 + 1
    if v2<1e-6 or v1<1e-6:
    # if v2 < 1e-20 :
        return 0
    else:
        return v1 / v2

def Euclidean(a,b):
    return np.linalg.norm(a - b)

def Jaccard(a,b):
    # 用Jaccard系数表示相关度
    return  a*b/(a*a+b*b-a*b)

distMeasure=1  #相似度计算方式
cal_similar={0:Euclidean,1:Cosin,2:Jaccard}

#量化误差quantization_error
def quantization_error(centroids, labels, data):
    error =0.0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist=0
        for sample in data[idx]:
            dist+=cal_similar.get(distMeasure)(sample,c)
        # print(dist,len(idx))
        dist /= len(idx)
        error += dist
    error /= len(centroids)
    return error

def reset_label(dist_matrix):
    count=0
    row_clust=[]
    while (True):
        row_clust = np.argmax(dist_matrix, axis=1)
        col_clust = np.argmax(dist_matrix, axis=0)
        id_dic={}
        label_set = set()
        #update_centrodis期望与target_centrodis对应的序号
        for i, label in enumerate(row_clust):
            if label not in label_set:
                id_dic[label]=[]
                label_set.add(label)
            id_dic[label].append(i)
        success = True
        for key in id_dic.keys():
            label_list=id_dic[key]
            #update_centrodis存在两个聚类中心期望对应target_centrodis同一个聚类中心
            if len(label_list) > 1 :
                success = False
                #update_centrodis被挑选中的聚类中心
                select_id=col_clust[key]
                #将被target_centrodis选中的select_id所在行的相似度置为最小值-1，防止再度参与竞争
                for i in range(len(dist_matrix)):
                    if i != key:
                        dist_matrix[select_id][i] = -1.1
                #将被target_centrodis选中的select_id所在列相似度置为最小值-1，表示update_centrodis其他聚类中心竞争失败，方便竞争target_centrodis其他聚类中心
                for i in range(len(dist_matrix)):
                    if i != select_id:
                        dist_matrix[i][key]=-1.1
        count+=1
        if success == True or count==10:
            break
    return success,row_clust

def reset_centroids(update_centrodis, target_centrodis,distanceMeasure=distMeasure):
    distances=np.array([[cal_similar.get(distanceMeasure)(a,b) for a in target_centrodis]for b in update_centrodis])
    # print("调整前的相似度矩阵\n",distances)
    success,idx = reset_label(distances)
    new_centroids=update_centrodis
    if success==True:
        index=[list(idx).index(i) for i in range(len(update_centrodis))]
        new_centroids =update_centrodis[index]
    # print("调整后\n",np.array([[cal_similar.get(distanceMeasure)(a,b) for a in target_centrodis]for b in update_centrodis])
    return new_centroids

def entropy(classes):
    label_total = []
    entropy_list = []
    entropy = 0
    for labels in classes:
        clust_label_sum = sum(labels)
        label_total.append(clust_label_sum)
        if clust_label_sum > 0:
            entropy_list.append(
                sum([0 - topic / clust_label_sum * math.log2(topic / clust_label_sum) for topic in labels]))
        else:
            entropy_list.append(0)
    doc_sum = sum(label_total)
    for i in range(len(label_total)):
        entropy += entropy_list[i] * label_total[i] / doc_sum
    return entropy

def purity(labels, clusters):
    matrix = confusion_matrix(clusters,labels)
    return np.max(matrix, axis=0).sum() / len(labels)
    label_sum = 0
    label_total = 0
    for labels in classes:
        label_sum += max(labels)
        label_total += sum(labels)
    return label_sum / label_total

#簇间距离
def dist_centroids(a,b,clusters,data):
    a_idx=np.where(clusters == a)
    b_idx=np.where(clusters == b)
    dist=[[np.linalg.norm(a-b) for a in data[a_idx]]for b in data[b_idx]]
    result=[min(lt) for lt in dist]
    return min(result)

#簇内距离
def centroids_inner(i,clusters,data):
    idx=np.where(clusters == i)
    dist=[[np.linalg.norm(a-b) for a in data[idx]]for b in data[idx]]
    result=[max(lt) for lt in dist]
    return max(result)

def Dunn_index(n_cluster,clusters,disXX):
    dis_outer=1e30
    dis_inner=-1e30
    for i in range(n_cluster):
        idx=np.where(clusters==i)
        if len(idx[0])==0:
            continue
        for j in range(i+1,n_cluster):
            vol=np.where(clusters==j)
            if len(vol[0])>0:
                temp=min([[disXX[a][b]for a in idx[0]]for b in vol[0]])
            dis_outer=min(dis_outer,temp)

    for i in range(n_cluster):
        vol = np.where(clusters == i)
        if len(vol[0]) > 0:
            temp=max([disXX[a][b] for a in vol[0] for b in vol[0] if b>a])
        dis_inner=max(dis_inner,temp)
    return dis_outer/dis_inner

def CH_index(n_cluster,clusters,centroids,data,meanX,distanceMeasure=distMeasure):
    # print(centroids,cal_disXC(centroids, data, distMeasure)[:10],clusters[:10])
    traceB=0
    traceW=0
    rcluN=0
    for i in range(n_cluster):
        idx=np.where(clusters==i)
        if len(idx[0])==len(data):
            return -1
        if len(idx[0])>0:
            traceB+=len(idx[0])*(cal_similar.get(distanceMeasure)(meanX, centroids[i])**2)
            rcluN+=1
            traceW+=np.sum([cal_similar.get(distanceMeasure)(obj,centroids[i])**2 for obj in data[idx]])
    # print(n_cluster, rcluN, traceB, traceW)
    CH=(traceB/(rcluN-1.0))/(traceW/(len(data)-rcluN))
    return CH

# def DB_index(clusters,data,centroids):
#     db_index=0
#     for i in range(len(centroids)):
#         R_list=[average_error(i,clusters,data,centroids)+average_error(j,clusters,data,centroids)
#                 /np.linalg.norm(centroids[i]-centroids[j]) for j in range(len(centroids)) if j!=i]
#         R=max(R_list)
#         db_index+=R
#     return db_index


def cal_disXC(centroids,data,distanceMeasure=distMeasure):
    disXC=[[cal_similar.get(distanceMeasure)(obj,cen) for cen in centroids]for obj in data]
    return np.array(disXC)

def cal_disXX(data,distanceMeasure=distMeasure):
    disXX=[[cal_similar.get(distanceMeasure)(a,b)for a in data]for b in data]
    return np.array(disXX)

def cal_fitness_disXC(disXC,distanceMeasure=distMeasure):
    if distanceMeasure==0:
        return np.sum(np.min(disXC,axis=1))
    else:
        return np.sum(np.max(disXC,axis=1))

def assign_cluster(centroids,data,distanceMeasure=distMeasure):
    disXC = cal_disXC(centroids, data, distanceMeasure)
    if distanceMeasure==0:
        clusters=np.argmin(disXC,axis=1)
    elif distanceMeasure==1:
        clusters=np.argmax(disXC,axis=1)
    return clusters

def update_centroids(n_cluster,clusters,data):
    centroids = []
    for i in range(n_cluster):
        idx = np.where(clusters == i)
        if len(idx[0]) == 0:
            id = int(np.random.uniform() * len(data))
            centroid = data[id]
            # centroid=random.choice(data)
        else:
            centroid = np.mean(data[idx],axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    return centroids

def update_centroids2(n_cluster,clusters,data):
    centroids = []
    for i in range(n_cluster):
        idx = np.where(clusters == i)
        print(idx)
        if not idx.any():
            centroid=random.choice(data)
        else:
            centroid = np.mean(data[idx],axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    return centroids

def update_centroids1(n_cluster,clusters,data,distanceMeasure=distMeasure):
    centroids = []
    mean_centroid=np.zeros_like(data[0])
    max_val=-1
    for i in range(n_cluster):
        idx = np.where(clusters == i)
        if len(idx[0]) == 0:
            centroid=random.choice(data)
        else:
            mean_centroid = np.mean(data[idx],axis=0)
            # print(idx[0])
            for j in idx[0]:
                dist=cal_similar.get(distanceMeasure)(mean_centroid,data[j])
                if dist>max_val:
                    max_val=dist
                    centroid=data[j]
        centroids.append(mean_centroid)
    centroids = np.array(centroids)
    return centroids

def cal_fitness(centroids,clusters,data,distanceMeasure=distMeasure):
    fitness=0
    for i, c in enumerate(centroids):
        idx=np.where(clusters == i)
        for sample in data[idx]:
            fitness+=cal_similar.get(distanceMeasure)(sample,c)
    # print(fitness)
    return fitness

def cal_fitness1(centroids,clusters,data,meanX,distanceMeasure=distMeasure):
    #ch = calinski_harabaz_score(data, clusters) 当有簇集为空时该函数会抛出异常
    return CH_index(len(centroids), clusters, centroids, data,meanX)

def cal_cluster_result(gbest_score,labels,clusters):
    ari = adjusted_rand_score(labels, clusters)
    v_m=v_measure_score(labels, clusters)
    fowlkes=fowlkes_mallows_score(labels, clusters)
    f1score=f1_score(labels, clusters, average='macro')
    mutu_info=mutual_info_score(labels, clusters)
    purt = purity(labels, clusters)
    cluster_result=[gbest_score,ari,v_m,fowlkes,f1score,mutu_info,purt]
    return cluster_result

class Clu:
    def __init__(self):
        self.centroid=[]
        self.volume=0
        self.objIndex=[]
        self.minDistanceVolume=0
        self.minDistanceObjIndex=[]

def assign_cluster1(centroids,data,disXX):
    # print("assign_version1")
    dataN=data.shape[0]
    clusterNum=len(centroids)
    clu=[]
    for _ in range(clusterNum):
        clu.append(Clu())
    set1=[i for i in range(dataN)]
    newAddObj=[]
    MinDistance=[]
    objClu=[]
    disXC = cal_disXC(centroids, data)
    # print(disXC)
    # print(disXX)
    # print(np.argmin(disXC,axis=1))
    clusters=[-1 for _ in range(dataN)]

    for obj in set1:
        c=np.argmin(disXC[obj])
        d=np.min(disXC[obj])
        MinDistance.append(d)
        objClu.append(c)
        clu[c].minDistanceObjIndex.append(obj)

    for i in range(clusterNum):
        v=len(clu[i].minDistanceObjIndex)
        # print(clu[i].minDistanceObjIndex)
        if v==0:
            continue
        MinDist=-1e30
        for obj in clu[i].minDistanceObjIndex:
            if MinDist<MinDistance[obj]:
                MinDist=MinDistance[obj]
                MinDisObject=obj
        newAddObj.append(MinDisObject)
        clu[i].objIndex.append(MinDisObject)
        clusters[MinDisObject]=i
        set1.remove(MinDisObject)
    # print(newAddObj,clusters)
    # print(objClu)
    while(len(set1)):
        for i in range(clusterNum):
            clu[i].minDistanceObjIndex=[]
        set1_copy = set1.copy()
        # print(newAddObj)
        # print(set1)
        for obj in set1:
            d=MinDistance[obj]
            c=objClu[obj]
            for nerst_obj in newAddObj:
                # print(obj,nerst_obj,disXX[obj][nerst_obj],clusters[nerst_obj])
                if disXX[obj][nerst_obj]>d:
                    d=disXX[obj][nerst_obj]
                    c=clusters[nerst_obj]
            MinDistance[obj]=d
            objClu[obj] = c
            clu[c].minDistanceObjIndex.append(obj)
        # print(objClu)
        newAddObj=[]
        temp_dis = -1e30
        obj_temp = -1
        for i in range(clusterNum):
            MinDist = -1e30
            v = len(clu[i].minDistanceObjIndex)
            # print("clu",i,clu[i].minDistanceObjIndex,clu[i].objIndex)
            if v == 0:
                continue
            for obj in clu[i].minDistanceObjIndex:
                if MinDist < MinDistance[obj]:
                    MinDist = MinDistance[obj]
                    MinDisObject = obj

            if MinDistance[MinDisObject]>temp_dis:
                temp_dis = MinDistance[MinDisObject]
                obj_temp = MinDisObject
        if obj_temp != -1:
            # print("obj_temp",obj_temp)
            newAddObj.append(obj_temp)
            clusters[obj_temp]=objClu[obj_temp]
            # obj[obj_temp]= objClu[obj_temp]
            # print(objClu[obj_temp])
            clu[objClu[obj_temp]].objIndex.append(obj_temp)
            set1.remove(obj_temp)
        # print("len set",len(set1),newAddObj)
        # set1=set1.copy
    # print("clusters",clusters,clusters.count(-1))
    return np.array(clusters)




















