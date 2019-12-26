import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.manifold import TSNE
from get_tfidf_matrix import get_data_matrix

def calc_index(file_path,index,res_type="mean"):
    df = pd.read_csv(file_path,sep=",")
    # print(df)
    data=df[index]
    if res_type=="mean":
        return round(data.mean(),3)
        return float("{:.3f}".format(data.mean(),3))
    elif res_type=="max":
        return round(data.max(), 3)
    elif res_type=="std":
        return round(data.std(), 3)
    elif res_type=="mean_std":
        return str("%.4f" % data.mean()) + "±" + str("%.4f" % data.std())
    else:
        return list(data)
#计算数据集的各个指标的max，mean,std值
def get_result(file_root,method_list,dataset_range,index_list,res_type):
    total_result = []
    for method in method_list:
        result = []
        for i in dataset_range:
            file_path = file_root + datasets[i] + "/" + method + "-" + "_result.csv"
            for index in index_list:
                if not os.path.exists(file_path):
                    print("文件{}不存在，结果用0代替".format(file_path))
                    res = 0
                else:
                    res = calc_index(file_path, index, res_type)
                    print(res)
                result.append(res)
        total_result.append(result)
    # print(np.array(total_result).T)
    index = pd.MultiIndex.from_product([[datasets[i] for i in dataset_range], index_list])
    df = pd.DataFrame(np.around(np.array(total_result).T, decimals=3), index=index,
                      columns=method_list)  # ,index=[datasets[0:5],index_list]
    print(df)
    df.to_csv(file_root + res_type + "-total_result.csv")
#画出标签分布的聚类分布散点图
def clusterDistributionMap():
    max_test_doc_num = 100
    low_words_num = 30
    up_words_num = 1e9
    is_shuff=False
    for i in range(0,5):
        tsne = TSNE()
        dataset_info,data,labels,n_cluster, corpus, itt, jtt,lmax,max_test_doc_num,low_words_num,topic,topic_docs=get_data_matrix(i,max_test_doc_num,low_words_num,up_words_num,is_shuff)
        # 进行数据降维,降成两维
        tsne.fit_transform(data)
        # label_file_path=file_root + datasets[i] + "/" + datasets[i] + "_labels.txt"
        # labels=pd.read_csv(label_file_path, header=None,sep='\t').values[:,0]
        tsne = pd.DataFrame(tsne.embedding_, index=labels)  # 转换数据格式
        fig = plt.figure()
        axes = fig.add_subplot(111)
        for c in range(n_cluster):
            d = tsne[labels == c]
            axes.scatter(d[0], d[1], color=plt.cm.Set1(c*1.0/ n_cluster),s=15,alpha=0.9)
        plt.show()


#计算p_value
def get_zval(data):
    zval_list=[]
    pvalue_list=[]
    for i in range(1,len(data)):
        rank_sum=stats.ranksums(data[i],data[0])
        zval=rank_sum[0]
        pvalue=rank_sum[1]
        zval_list.append(zval)
        pvalue_list.append(pvalue)
    return zval_list,pvalue_list

def p_value_test(id,others_id_list,index_list,method_list):
    z_val_list = []
    p_val_list = []
    for i in dataset_range:
        file_path_list = []
        file_path_list.append(file_root + datasets[i] + "/" + method_list[id] + "_result.csv")
        for j in others_id_list:
            file_path_list.append(file_root + datasets[i] + "/" + method_list[j] + "_result.csv")
        for index in index_list:
            result = []
            for file_path in file_path_list:
                if not os.path.exists(file_path):
                    print("文件{}不存在，结果用0代替".format(file_path))
                    res = []
                else:
                    res = calc_index(file_path, index, res_type="all")
                    # print(res)
                result.append(res)
            # print(np.array(result))
            zval, pvalue = get_zval(result)
            # print("pvalue",pvalue)
            z_val_list.append(zval)
            p_val_list.append(pvalue)
    print(np.array(p_val_list))
    index1=[datasets[i] for i in dataset_range]
    columns=[method_list[j] for j in others_id_list]
    index = pd.MultiIndex.from_product([index1, index_list], names=['数据集', '统计指标'])
    pvalue_result = pd.DataFrame(np.array(p_val_list), index=index,columns=columns)
    print(pvalue_result)
    pd.DataFrame(pvalue_result).to_csv(file_root  + "pvalue_result.csv")
    pd.DataFrame(np.array(pvalue_result)<0.05).to_csv(file_root+"pvalue_result_judge.csv")

#收敛图
def get_iter_result(file_root,method_list,dataset_range,index_list):
    total_result = []
    for i in dataset_range:
        file_path = file_root + datasets[i] + "/" +datasets[i] + "_opt_iter.csv"
        df = pd.read_csv(file_path)
        print(df)
        for j in method_list:
            print(df[method_list[j]])

#箱线图
def box_test(index_list,lt):
    for i in dataset_range:
        file_path_list=[]
        for j in lt:
            file_path_list.append(file_root + datasets[i] + "/" + method_list[j] + "_result.csv")
        for index in index_list:
            result = []
            for file_path in file_path_list:
                if not os.path.exists(file_path):
                    print("文件{}不存在，结果用0代替".format(file_path))
                    res = []
                else:
                    res = calc_index(file_path, index, res_type="all")
                    # print(res)
                result.append(res)
            pd.DataFrame(np.array(result).T,columns=[method_list[j] for j in lt]).to_csv(file_root + datasets[i] +"_"+index+".csv")

def total_iter(file_root,dataset,method_list=['PSO', 'ACI-PSO', 'DE', 'ACI-DE']):
    file_path = file_root + dataset + "/" + dataset+ "_opt_iter.csv"
    file_path1 = file_root + dataset + "/" + dataset+ "_mean_iter.csv"
    df = pd.read_csv(file_path)
    df1=pd.read_csv(file_path1)
    for method in method_list:
        arr=df[method].values
        np.savetxt(file_root +dataset + "/"+method+ "_opt_iter.txt", arr, delimiter="\t", fmt="%.3f")
        arr1 = df1[method].values
        np.savetxt(file_root + dataset + "/" + method+"_mean_iter.txt", arr1, delimiter="\t", fmt="%.3f")

method_list=["K-means","K-means++","PSO","DE","ACI-PSO","ACI-DE","KDEPSO","ACI-KDEPSO"] #"KQPSO","KDEPSO"
method_dic={0:"K-means",1:"k-means++",2:"PSO",3:"CLPSO",4:"DE",5:"QPSO",6:"PSO-DE",7:"DE-PSO"}
datasets=["topics_webkb4","topics_r5","topics_r8","topics_r10","20newsgroup","20ng-t6","20ng-t15"]
max_iter_list=[60,60,100,100,150,100,150]  #每个数据集的迭代次数
datasets1 = ["three_Ren","half_ring","two-rings","RING-GAUSSIAN",   #非凸结构数据集
             "4guass","iris","wine","WisconsinBreastCancer",        #UCI数据集
             "2d-4c-no8","2d-10c-no2","10d-4c-no1","ellipsoid.50d4c.9",
              "ellipsoid.100d4c.9","2d-20c-no0",                    # 9-14各指标方差接近0的数据集
              "10d-10c-no0","10d-20c-no9","ellipsoid.50d10c.9","ellipsoid.100d10c.9",  #15-18
              "zoo","wdbc","ionosphere","lung-cancer","sonar","movement_libras","Hill_Valley",
              "Musk version1","arrhythmia","madelon","isolet5"]
cluster_index=["F","ARI","v_measure","FMI","f1-measure","MI","purity","CH","run_time"]
file_root="C:/Users/Administrator/Desktop/Note/wmf/10-30/PSO-2/"           #PSOTC(100-30)/
# file_root=r"dataset/"
dataset_range=range(0,5)  #数据集的测试范围
id=4
others_id_list=[0,1,2]
index_list=cluster_index[:4]
p_value_test(id,others_id_list,index_list,method_list)

# box_test(index_list[:2],[0,1,2,4])
#收敛曲线
#get_iter_result(file_root,method_list,dataset_range,index_list)

# for i in range(0,4):
#     print(datasets[i])
#     total_iter(file_root, datasets[i])





