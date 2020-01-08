import random
import numpy as np
import pandas as pd
import time
import os

import shutil
from sklearn.cluster import KMeans
from index import reset_centroids, purity, update_centroids, assign_cluster, cal_fitness, CH_index
from sklearn.metrics import *
from sklearn.metrics import f1_score
from get_tfidf_matrix import get_data_matrix,get_uci_data
from clusterAlgorithm import Clustering
from kmeans import Kmeans
from particle import Particle


def opt_algo(pathrr,n_cluster,data,labels,method_id,data_id,use_ACI,use_kmeans,trail,n_particles,max_iter):
    print("cluster result by {} of {},是否使用ACI操作{}:".format(method_dic[method_id],datasets[data_id],use_ACI))
    max_fitness = -np.inf
    iter_result_mean=[]
    gbest_cluster_result=[]
    opt_cluster=[]
    for i in range(trail):
        start = time.perf_counter()
        if method_id<2:
            if method_id==0:
                km = Kmeans(n_cluster=n_cluster, init_pp=False,max_iter=max_iter)
            elif method_id==1:
                km = Kmeans(n_cluster=n_cluster, init_pp=True,max_iter=max_iter)
            centroid = km._init_centroid(data)
            gbest_fitness, clusters, centroids, iter_result= km.fit(data, centroid)
        else:
            clusterSwarm = Clustering(n_cluster=n_cluster, n_particles=n_particles, data=data, labels=labels,
                                      use_kmeans=use_kmeans, max_iter=max_iter,w=w)
            if method_id == 2:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.pso_run(w=w, c1=c1, c2=c2,use_ACI=use_ACI)
            elif method_id == 3:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.clpso_run(w=w, c=c1,use_ACI=use_ACI)
            elif method_id == 4:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.de_run(CR=cr, F=F,use_ACI=use_ACI)
            elif method_id == 5:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.qpso_run(alpha=alpha,use_ACI=use_ACI)
            elif method_id == 6:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.de_pso_run(use_ACI=use_ACI,w=w, c1=c1, c2=c2, CR=cr, F=F)
            elif method_id == 7:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.de_qpso_run(alpha=alpha, CR=cr, F=F,use_ACI=use_ACI)
            elif method_id==8:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.pso_run_Subpopulation(w=w, c1=c1, c2=c2,pro=pro,use_ACI=use_ACI)
            elif method_id == 9:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.de_run_Subpopulation(CR=cr, F=F, pro=pro,use_ACI=use_ACI)
            elif method_id == 10:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.pso_de_run_Subpopulation(w=w, c1=c1, c2=c2,CR=cr, F=F, pro=pro,
                                                                                                    use_ACI=use_ACI)
            elif method_id == 11:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.de_run(CR=cr, F=F,use_ACI=use_ACI)
                clusterSwarm = Clustering(n_cluster=n_cluster, n_particles=n_particles, data=data, labels=labels,
                                      use_kmeans=use_kmeans, max_iter=max_iter,w=w)
                #将DE的聚类结果初始化PSO种群的一个个体
                pop=Particle(n_cluster, data,labels,use_kmeans)
                # print(clusters)
                for i in range(n_cluster):
                    clusters=[]
                    clusterSwarm.particles[i].centroids = pop.init_centroids3(data, n_cluster, clusters)
                    clusterSwarm.particles[i].clusters = assign_cluster(clusterSwarm.particles[i].centroids, data)
                    clusterSwarm.particles[i].fitness = cal_fitness(clusterSwarm.particles[i].centroids,
                                                                    clusterSwarm.particles[i].clusters, data)
                # clusterSwarm.particles[0].centroids=centroids.copy()
                # clusterSwarm.particles[0].cluster=clusters.copy()
                # clusterSwarm.particles[0].fitness=gbest_fitness
                clusterSwarm._update_gbest()
                # print(gbest_fitness,clusterSwarm.gbest_fitness,clusterSwarm.particles[0].best_fitness)
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.pso_run(w=w, c1=c1, c2=c2,use_ACI=use_ACI)
            elif method_id == 12:
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.de_run(CR=cr, F=F,use_ACI=use_ACI)
                clusterSwarm = Clustering(n_cluster=n_cluster, n_particles=n_particles, data=data, labels=labels,
                                      use_kmeans=use_kmeans, max_iter=max_iter,w=w)
                #将DE的聚类结果初始化QPSO种群的一个个体
                pop=Particle(n_cluster, data,labels,use_kmeans)
                for i in range(n_cluster):
                    clusterSwarm.particles[i].centroids = pop.init_centroids3(data, n_cluster, clusters)
                    clusterSwarm.particles[i].clusters = assign_cluster(clusterSwarm.particles[i].centroids, data)
                    clusterSwarm.particles[i].fitness = cal_fitness(clusterSwarm.particles[i].centroids,
                                                                    clusterSwarm.particles[i].clusters, data)
                clusterSwarm._update_gbest()
                gbest_fitness, clusters, centroids, iter_result = clusterSwarm.qpso_run(alpha=alpha,use_ACI=use_ACI)

        #将样本聚类后的标签和聚类中心进行调整
        opt_centroids = update_centroids(n_cluster, labels, data)
        centroids = reset_centroids(centroids,opt_centroids)
        clusters = assign_cluster(centroids, data)
        # print(confusion_matrix(clusters, labels))
        max_val=iter_result[len(iter_result) - 1]
        while len(iter_result) < max_iter:
            iter_result.append(max_val)

        #每次重复运行后各指标的最优值统计
        iter_result_mean.append(iter_result)
        if max_fitness < gbest_fitness:
            max_fitness = gbest_fitness
            iter_result_best = iter_result
            opt_cluster = clusters
        # print(labels,clusters)
        index_result=[gbest_fitness,adjusted_rand_score(labels, clusters),v_measure_score(labels,clusters),
                      fowlkes_mallows_score(labels,clusters),f1_score(labels, clusters, average='macro'),
                      mutual_info_score(labels, clusters),purity(labels,clusters),
                      CH_index(n_cluster, clusters, centroids, data,np.mean(data,axis=0)),
                      (time.perf_counter()-start)]
        gbest_cluster_result.append(index_result)
        # print(i,index_result)
    df=pd.DataFrame(np.around(np.array(gbest_cluster_result), decimals=3),columns=cluster_index)
    print(df.round(3))
    print(df.describe().round(3))
    iter_result_mean = np.mean(np.array(iter_result_mean), axis=0)
    #每次独立运行后以最优聚类中心计算的各指标
    if method_id<2 or use_ACI==False:
        prefix_path=pathrr+method_dic[method_id]
    else:
        prefix_path = pathrr + "ACI-" + method_dic[method_id]
    df.to_csv(prefix_path +"_result.csv",mode='a')
    df.describe().to_csv(prefix_path+"_stat.csv")
    np.savetxt(prefix_path+"_opt_cluster.txt", np.array(opt_cluster), delimiter="\t", fmt="%.0f")
    np.savetxt(prefix_path+"_opt_iter.txt", iter_result_best, delimiter="\t", fmt="%.3f")
    np.savetxt(prefix_path+"_mean_iter.txt", iter_result_mean, delimiter="\t", fmt="%.3f")

def del_dir(path):
    ls=os.list(path)
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_dir(c_path)
        else:
            os.remove(c_path)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('目录../{}已建立！'.format(path))

def calc_index(file_path,index,res_type="mean"):
    df = pd.read_csv(file_path,sep=",")
    # print(df)
    data=df[index]
    if res_type=="mean":
        return round(data.mean(),3)
        # return float("{:.3f}".format(data.mean(),3))
    elif res_type=="max":
        return round(data.max(), 3)
    elif res_type=="std":
        return round(data.std(), 3)
    else:
        return str("%.3f" % data.mean()) + "±" + str("%.3f" % data.std())

def get_result(file_root,method_list,dataset_range,index_list,res_type):
    total_result = []
    for method in method_list:
        result = []
        for i in dataset_range:
            file_path = file_root + datasets[i] + "/" + method  + "_result.csv"
            for index in index_list:
                if not os.path.exists(file_path):
                    print("文件{}不存在，结果用0代替".format(file_path))
                    res = 0
                else:
                    res = calc_index(file_path, index, res_type)
                    # print(res)
                result.append(res)
        total_result.append(result)
    # print(np.array(total_result).T)
    index = pd.MultiIndex.from_product([[datasets[i] for i in dataset_range], index_list])
    df = pd.DataFrame(np.around(np.array(total_result).T, decimals=3), index=index,
                      columns=method_list)  # ,index=[datasets[0:5],index_list]
    print(df)
    df.to_csv(file_root + res_type + "-total_result.csv")

def total_iter(file_root,dataset,method_list):
    opt_iter_list=[]
    mean_iter_list=[]
    for method in method_list:
        file_path=file_root+dataset+"/"+method+ "_opt_iter.txt"
        opt_iter_list.append(np.loadtxt(file_path))
        file_path = file_root + dataset + "/" +method + "_mean_iter.txt"
        mean_iter_list.append(np.loadtxt(file_path))

    pd.DataFrame(np.around(np.array(opt_iter_list).T, decimals=3), columns=method_list) \
        .to_csv(file_root + dataset + "/" + dataset + "_opt_iter.csv")
    pd.DataFrame(np.around(np.array(mean_iter_list).T, decimals=3), columns=method_list) \
        .to_csv(file_root + dataset + "/" + dataset + "_mean_iter.csv")

def get_total_result():
    method_list = ["K-means", "K-means++", "PSO", "ACI-PSO","DE", "ACI-DE","DE-subpop", "ACI-DE-subpop", "DE-PSO",
                   "ACI-DE-PSO"]  # "KQPSO","KDEPSO"
    file_root="C:/Users/Administrator/Desktop/Note/wmf/1-1/QPSO-1/"
    # file_root="C:/Users/Administrator/Desktop/Note/wmf/12-12//"
    m_list=["QPSO", "ACI-QPSO"]
    dataset_range=range(0,5)
    get_result(file_root, m_list, dataset_range, cluster_index[:4], "mean")
    get_result(file_root, m_list, dataset_range, cluster_index[:4], "std")
    get_result(file_root, m_list, dataset_range, cluster_index[:4], "max")


#PSO参数：惯性权重(w),学习因子（c1,c2）
w=0.9
c1=1.495
c2=1.495
#DE参数：收缩因子(F),交叉概率（c1,c2）
F=2.0
cr=0.7
pro=0.5
#QPSO参数：收缩膨胀系数(alpha)
alpha=0.9
method_list=["K-means","K-means++","KPSO","KDE","ACI-PSO","ACI-DE","KDEPSO","ACI-KDEPSO"] #"KQPSO","KDEPSO"
method_dic={0:"K-means",1:"k-means++",2:"PSO",3:"CLPSO",4:"DE",5:"QPSO",6:"PSO-DE",7:"DE-QPSO",
            8:"PSO-subpop",9:"DE-subpop",10:"PSO-DE-subpop",11:"DE-PSO",12:"GQPSO"}
datasets=["topics_webkb4","topics_r5","topics_r8","topics_r10","20newsgroup","20ng-t6","20ng-t15"]
max_iter_list=[80,80,80,100,150,100,150]  #每个数据集的迭代次数
datasets1 = ["three_Ren","half_ring","two-rings","RING-GAUSSIAN",   #非凸结构数据集
             "4guass","iris","wine","WisconsinBreastCancer",        #UCI数据集
             "2d-4c-no8","2d-10c-no2","10d-4c-no1","ellipsoid.50d4c.9",
              "ellipsoid.100d4c.9","2d-20c-no0",                    # 9-14各指标方差接近0的数据集
              "10d-10c-no0","10d-20c-no9","ellipsoid.50d10c.9","ellipsoid.100d10c.9",  #15-18
              "zoo","wdbc","ionosphere","lung-cancer","sonar","movement_libras","Hill_Valley",
              "Musk version1","arrhythmia","madelon","isolet5"]
cluster_index=["F","ARI","v_measure","FMI","f1-measure","MI","purity","CH","run_time"]
file_root=r"dataset/"
dataset_range=range(3)     #数据集的测试范围
method_range=[10]     #测试的算法范围
######################################
def main(dataset_range,trail=20,n_particles=40,use_kmeans=False,is_shuff=False):
    for data_id in dataset_range:
        print("test dataset: ",datasets[data_id])
        max_test_doc_num=100
        low_words_num=30
        up_words_num=1e9
        # max_iter=max_iter_list[data_id]
        max_iter=150
        file_rootrr="result" + str(max_test_doc_num) + "-" + str(low_words_num) + "/"
        # shutil.rmtree(file_rootrr)
        pathrr = file_rootrr + datasets[data_id]+"/"
        mkdir(pathrr) ######################################
        #text datasets
        dataset_info,data,labels,n_cluster=get_data_matrix(data_id,max_test_doc_num,low_words_num,up_words_num,is_shuff)
        # UCI datasets
        # data,labels,n_cluster=get_uci_data(data_id,datasets)
         ######################################
        # print(data[:2],"\n labels(0-50):\n",labels[:50])
        # print(data.max(axis=0),"\n",data.min(axis=0),"\n",data.mean(axis=0))
        # print("std of data\n",np.std(data,axis=0))
        # storage labels, opt_centroids and tfidf data_matrix
        # np.savetxt(pathrr  +"labels.txt", labels, delimiter="\t",fmt="%.1f")
        opt_centroids = update_centroids(n_cluster, labels, data)
        # pd.DataFrame(opt_centroids).to_csv(pathrr + "centroids_ori.csv",index=False) #no index row
        # pd.DataFrame(data).to_csv(pathrr + "data.csv", index=False)  # no index row data samples

        ######################################
        dataset_info["test method"] = [method_dic[i] for i in method_range]
        dataset_info["test method"] = [method_dic[i] for i in method_range]
        dataset_info["PSO"] = ["w="+str(w),"c1=c2="+str(c1),"pro="+str(pro)]
        dataset_info["DE"] = ["F="+str(F),"cr="+str(cr)]
        dataset_info["trail"] = trail
        dataset_info["n_particles"] = n_particles
        dataset_info["max_iter"] = max_iter
        dataset_info["CH-index of orginal clusters"]=calinski_harabaz_score(data, labels)
        dataset_info["fitness of orginal clusters"]=cal_fitness(opt_centroids, labels, data)

        mylog = open(pathrr + "info.txt", mode='w', encoding='utf-8')
        for key, val in dataset_info.items():
            print(key + ": " + str(val))
            mylog.write(key + ": " + str(val) + '\n')
        mylog.close()
        ######################################
        method_list=[]
        for method_id in method_range:
            use_ACI = False
            method_list.append(method_dic[method_id])
            opt_algo(pathrr,n_cluster,data,labels,method_id,data_id,use_ACI,use_kmeans,trail,n_particles,max_iter)

            if method_id>=2:
                use_ACI = True
                method_list.append("ACI-"+method_dic[method_id])
                opt_algo(pathrr,n_cluster, data, labels, method_id, data_id,use_ACI,use_kmeans,trail,n_particles,max_iter)

    # get the opt/men iter and mean,std,max of total result
    total_iter(file_rootrr, datasets[data_id], method_list)
    get_result(file_rootrr,method_list, dataset_range, cluster_index[:4], res_type="mean")
    get_result(file_rootrr,method_list, dataset_range, cluster_index[:4], res_type="std")
    get_result(file_rootrr,method_list, dataset_range, cluster_index[:4], res_type="max")

if __name__ == "__main__":
    print("test start time：",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # get_total_result()

    main(dataset_range)
    print("test end time：",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))












