import random

from datetime import datetime
import numpy as np
import pandas as pd
import math

from sklearn.cluster import KMeans
# from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier

from clusterAlgorithm import Clustering
from get_tfidf_matrix import get_uci_data
from index import update_centroids, reset_centroids, assign_cluster

# def update_gbest(size,fitness,pbest_fitness,gbest_fitness):
#     gbest_index=0
#     for i in range(size):
#         if fitness[i]>pbest_fitness[i]:
#             pbest_fitness[i]=fitness[i]
#             PBest[i]=pos[i]
#
#         if fitness[i]>gbest_fitness:
#             gbest_fitness=fitness[i]
#             gbest_index=i
#     GBest=pos[gbest_index]
#     return GBest
# random = np.random.RandomState(0)  # 随机数种子，相同种子下每次运行生成的随机数相同
def Rand(low,up):
    val = random.uniform(low, up)
    return val
    # return random.random(low,up)

def bound(x,low=-1,up=1):
    if x<low:
        x=low
    elif x>up:
        x=up
    return x

def convert_binary_pos(pos):
    return np.array([0 if val < 0.5 else 1 for val in pos])

def feature_select(data, pos):
    data_redu=data.copy()
    idx = np.where(pos > 0.5)  # [i for i in range(data.shape[1]) if pos[i]>0.5]
    if len(idx[0]) > 0:
        data_redu = data_redu[:, idx[0]]
    # print(data.shape[1],data_redu.shape[1])
    return data_redu

def LDA(X, y):
    lda = LinearDiscriminantAnalysis(n_components=None)
    lda.fit(X, y)
    clusters = lda.predict(X)
    accuarcy = lda.score(X, y)
    return accuarcy

def neigh(X, Y):
    knn = KNeighborsClassifier(n_neighbors=5)
    # X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.3, random_state=0)  # train_test_split
    accuarcy = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    # knn.fit(X_test,Y_test)
    # a=knn.score(X_test,Y_test)
    # print(a)
    # X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.3, random_state=0)  # train_test_split
    # knn.fit(X,Y)
    # knn.fit(X_test,Y_test)
    # score=knn.score(X,Y)
    # return score
    # print("accuarcy\t",accuarcy.mean())
    # neigh.fit(X, y)
    # accuarcy = neigh.score(X, y)
    return accuarcy.mean() #accua

def cal_dim(feature_set):
    count=0
    for f in feature_set:
        if f>=0.5:
            count+=1
    return count

def PSO_cluster(data, labels, w=0.8, c1=2.0, c2=2.0, use_ACI=True, trail=2):
    n_cluster = len(np.unique(labels))
    # print(n_cluster)
    ari = 0
    for i in range(trail):
        clusterSwarm = Clustering(n_cluster=n_cluster, n_particles=20, data=data, labels=labels,
                                  use_kmeans=False)  # max_iter=200, print_debug=50)
        gbest_score, clusters, centroids, iter_result, cluster_result = clusterSwarm.pso_run(w=w, c1=c1, c2=c2,use_ACI=use_ACI)
        opt_centroids = update_centroids(n_cluster, labels, data)
        centroids = reset_centroids(centroids, opt_centroids)
        clusters = assign_cluster(centroids, data)
        ari += adjusted_rand_score(labels, clusters)
    return ari / trail

def Kmeans_cluster(data, labels,trail):
    n_cluster = len(np.unique(labels))
    # print(n_cluster)
    ari_list = []
    for i in range(trail):
        km_cluster = KMeans(n_clusters=n_cluster, max_iter=100, n_init=10, init='k-means++', n_jobs=1)
        clusters = km_cluster.fit_predict(data)
        centroids = km_cluster.cluster_centers_
        ari = adjusted_rand_score(clusters,labels)
        ari_list.append(ari)
    # print(ari_list)
    return np.mean(ari_list)

def cal_fitness0(data,labels,pos):
    binary_pos = convert_binary_pos(pos)
    # print(binary_pos)
    X=feature_select(data, binary_pos)
    # print(X[:10])
    # print(cal_dim(pos),X.shape)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, labels)
    accuarcy = neigh.score(X, labels)
    return accuarcy #LDA(X,labels)

def cal_fitness1(data,labels,pos):
    binary_pos = convert_binary_pos(pos)
    # print(binary_pos)
    X=feature_select(data, binary_pos)
    # print(X[:10])
    # print(cal_dim(pos),X.shape)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, labels)
    accuarcy = neigh.score(X, labels)
    return accuarcy #LDA(X,labels)

def cal_fitness2(data,labels,pos):
    binary_pos = convert_binary_pos(pos)
    # print(binary_pos)
    X = feature_select(data, binary_pos)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X,labels)
    accuarcy=knn.score(X,labels)#LDA(X,labels)
    # print(accuarcy)
    # errorRate=1-accuarcy
    # fitness=errorRate+alpha*cal_dim(pos)
    # print(cal_dim(pos),accuarcy)
    return accuarcy

def cal_fitness3(data,labels,pos,trail=1,alpha=1e-6):
    binary_pos = convert_binary_pos(pos)
    # print(binary_pos)
    X = feature_select(data, binary_pos)
    ari=Kmeans_cluster(X, labels,trail)
    return ari

def cal_fitness4(data,labels,pos,alpha=1e-6):
    binary_pos = convert_binary_pos(pos)
    # print(binary_pos)
    X = feature_select(data, binary_pos)
    ari=PSO_cluster(X, labels)
    return ari

cal_fitness={0:cal_fitness0,1:cal_fitness1,2:cal_fitness2,3:cal_fitness3,4:cal_fitness4}
fitness_measure=2
class FSPSO:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 size:int=20,
                 c1:int=2.0,
                 c2:int=2.0,
                 w:int=0.8,
                 max_iter:int=100):
        # X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=0)
        self.size=size
        self.data=data     #X_train
        self.labels=labels #Y_train
        self.n_samples=self.data.shape[0]
        self.dim=self.data.shape[1]
        self.FS_dim=self.data.shape[1]
        self.c1=c1
        self.c2=c2
        self.w=w
        self.max_iter=max_iter
        self.pos=[]
        self.vel=[]
        self.fitness=[]
        self.pbest_pos=[]
        self.pbest_fitness=[]
        self.gbest_pos=[Rand(0.5,1) for _ in range(self.dim)]
        self.gbest_fitness=None
        self.gbest_cluster=None
        self.gbest_centroids=None
        self.clf_accuarcy=0
        self.NO_FS_fitness=cal_fitness.get(fitness_measure)(data,labels,self.gbest_pos)
        self.update_gbest = {1:self.update_gbest1, 2:self.update_gbest2, 3: self.update_gbest3, 4: self.update_gbest4}
        self.init = {1:self._init, 2:self._small_init, 3: self._large_init, 4: self._mix_init}
        self.gbest_measure=1
        self.init_measure=1
        # original_accuarcy=neigh(self.data,self.labels)
        # print("original_accuarcy",original_accuarcy)
        # self.data_test=X_test
        # self.labels_test=Y_test

    def _init(self):
        self.pos = [[random.random() for _ in range(self.dim)] for i in range(self.size)]
        # print(self.pos[0])

    def _small_init(self):
        for i in range(self.size):
            idx = np.random.choice(range(self.dim), size=int(self.dim*0.1))
            x=[]
            for j in range(self.dim):
                if j in idx:
                    x.append(Rand(0.5,1))
                else:
                    x.append(Rand(0,0.5))
            self.pos.append(x)
        self.pos = [[random.random() for _ in range(self.dim)] for i in range(self.size)]

    def _large_init(self):
        for i in range(self.size):
            idx = np.random.choice(range(self.dim), size=int(self.dim*Rand(0.5,1)))
            x=[]
            for j in range(self.dim):
                if j in idx:
                    x.append(Rand(0.5,1))
                else:
                    x.append(Rand(0,0.5))
            self.pos.append(x)

    def _mix_init(self):
        small_size=int(self.size/3)
        for i in range(small_size):
            idx = np.random.choice(range(self.dim), size=int(self.dim * 0.1))
            x = []
            for j in range(self.dim):
                if j in idx:
                    x.append(Rand(0.5, 1))
                else:
                    x.append(Rand(0, 0.5))
            self.pos.append(x)

        for i in range(small_size,self.size):
            idx = np.random.choice(range(self.dim), size=int(self.dim * Rand(0.5, 1)))
            x = []
            for j in range(self.dim):
                if j in idx:
                    x.append(Rand(0.5, 1))
                else:
                    x.append(Rand(0, 0.5))
            self.pos.append(x)


    def _init_particle(self):
        self.init.get(self.init_measure)()
        self.vel=[[random.random() for _ in range(self.dim)]for i in range(self.size)]
        self.fitness=[cal_fitness.get(fitness_measure)(self.data,self.labels,self.pos[i]) for i in range(self.size)]
        # print(self.data[:10])
        # print(self.pos)
        # print(self.fitness)
        self.pbest_pos=self.pos.copy()
        self.pbest_fitness=self.fitness.copy()
        self.gbest_pos=self.pos[0]
        self.gbest_fitness=self.fitness[0]
        self.update_gbest.get(self.gbest_measure)()

    def update_gbest1(self):
        for i in range(self.size):
            # print(cal_dim(self.pos[i]))
            # print(self.pbest_fitness[i])
            if self.fitness[i] > self.pbest_fitness[i]:
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_pos[i] = self.pos[i]

            if self.fitness[i] > self.gbest_fitness:
                self.gbest_fitness = self.fitness[i]
                self.gbest_pos = self.pos[i]

    def update_gbest2(self):
        for i in range(self.size):
            if self.fitness[i] > self.pbest_fitness[i]:
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_pos[i] = self.pos[i]
            elif self.fitness[i] ==self.pbest_fitness[i] and cal_dim(self.pos[i])<cal_dim(self.pbest_pos[i]):
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_pos[i] = self.pos[i]

            if self.fitness[i] > self.gbest_fitness:
                self.gbest_fitness = self.fitness[i]
                self.gbest_pos=self.pos[i]

            elif self.fitness[i] ==self.gbest_fitness and cal_dim(self.pos[i])<cal_dim(self.gbest_pos):
                self.gbest_fitness = self.fitness[i]
                self.gbest_pos = self.pos[i]

    def update_gbest3(self):
        index = 0
        for i in range(self.size):
            if self.fitness[i] > self.pbest_fitness[i] and cal_dim(self.pos[i]) <=cal_dim(self.pbest_pos[i]):
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_pos[i] = self.pos[i]
            elif self.fitness[i] == self.pbest_fitness[i] and cal_dim(self.pos[i]) < cal_dim(self.pbest_pos[i]):
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_pos[i] = self.pos[i]

            if self.fitness[i] > self.gbest_fitness and cal_dim(self.pos[i]) <= cal_dim(self.gbest_pos):
                self.gbest_fitness = self.fitness[i]
                self.gbest_pos = self.pos[i]

            elif self.fitness[i] == self.gbest_fitness and cal_dim(self.pos[i]) < cal_dim(self.gbest_pos):
                self.gbest_fitness = self.fitness[i]
                self.gbest_pos = self.pos[i]

    def update_gbest4(self):
        index = 0
        for i in range(self.size):
            if self.fitness[i] > self.pbest_fitness[i] and cal_dim(self.pos[i]) <=cal_dim(self.pbest_pos[i]):
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_pos[i] = self.pos[i]
            elif self.fitness[i] > 0.95*self.pbest_fitness[i] and cal_dim(self.pos[i]) < cal_dim(self.pbest_pos[i]):
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_pos[i] = self.pos[i]

            if self.fitness[i] > self.gbest_fitness and cal_dim(self.pos[i]) <= cal_dim(self.gbest_pos):
                self.gbest_fitness = self.fitness[i]
                self.gbest_pos = self.pos[i]

            elif self.fitness[i] > 0.95*self.gbest_fitness and cal_dim(self.pos[i]) < cal_dim(self.gbest_pos):
                self.gbest_fitness = self.fitness[i]
                self.gbest_pos = self.pos[i]

    def update_pos(self,pc=0.5):
        for i in range(self.size):
            for j in range(self.dim):
                self.vel[i][j] = self.w * self.vel[i][j] + self.c1 * Rand(0,1) * (self.pbest_pos[i][j] - self.pos[i][j]) + self.c2 * Rand(0,1) * (
                    self.gbest_pos[j] - self.pos[i][j])
                self.vel[i][j] = bound(self.vel[i][j])
                #第一种更新方式
                self.pos[i][j] += self.vel[i][j]
                self.pos[i][j] = bound(self.pos[i][j], low=0, up=1)
                # # 第二种更新方式
                # self.pos[i][j]=1 if self.vel[i][j]<pc else 0
                # # 第三种更新方式
                # s=1/(1+math.exp(-self.vel[i][j]))
                # self.pos[i][j] = 1 if Rand(0,1) < s else 0

    def crossover_oprateor(self,CR=0.9):
        for i in range(self.size):
            for j in range(self.dim):
                if Rand(0,1)<CR:
                    a = random.sample(range(self.size), 1)
                    while a==i:
                        a = random.sample(range(self.size), 1)
                    self.pos[i][j]=self.pos[a][j]

    def muta_operator(self,mu=0.2):
        for i in range(self.size):
            nbest=i
            for j in range(self.dim):
                if Rand(0,1)<mu:
                    a = random.sample(range(self.size), 1)
                    b = random.sample(range(self.size), 1)
                    while a == i or b==i:
                        a = random.sample(range(self.size), 1)
                        a = random.sample(range(self.size), 1)
                    self.pos[i][j] = self.pos[nbest][j]+Rand(0,1)*(self.pos[a][j]-self.pos[b][j])

    def update_pos1(self,pc=0.5):
        # 第四种更新方式
        m=0.5
        r=1
        S0={}
        S1={}
        p=[]
        for i in range(self.dim):
            cnt0 = []
            cnt1 = []
            for j in range(self.size):
                if(self.pos[j][i]==0):
                    cnt0.append(j)
                else:
                    cnt1.append(j)

            S0[i]=cnt0
            S1[i]=cnt1

        for i in range(self.size):
            for j in range(self.dim):
                if len(S0[j])==0:
                    self.vel[i][j]=m*r
                elif len(S1[j])==0:
                    self.vel[i][j]=1-m*r
                else:
                    pass

    def update_iter(self):
        self.update_pos()
        for i in range(self.size):
            # print(binary_pos,self.pos[i])
            self.fitness[i] = cal_fitness.get(fitness_measure)(self.data, self.labels, self.pos[i])
            self.update_gbest.get(self.gbest_measure)()
            # print(i,cal_dim(self.gbest_pos),self.gbest_fitness)
    def run(self):
        self._init_particle()
        for t in range(self.max_iter):
            self.update_iter()
            self.FS_dim = cal_dim(self.gbest_pos)
            # print("iter:{:d},No_FS ARI:{:.4f} ARI:{:.4f},dim: {:d}".
            #       format(t, self.NO_FS_fitness, self.gbest_fitness, cal_dim(self.gbest_pos)))

dataset_list = [
        "three_Ren",
        "half_ring",
        "two-rings",
        "RING-GAUSSIAN",
        "4guass",  # 4
        "iris",
        "wine",
        "WisconsinBreastCancer",
        "2d-4c-no8",  # 8---400
        "2d-10c-no2",
        "10d-4c-no1",
        "ellipsoid.50d4c.9",
        "ellipsoid.100d4c.9",
        "2d-20c-no0",  # 13
        "10d-10c-no0",
        "10d-20c-no9",
        "ellipsoid.50d10c.9",
        "ellipsoid.100d10c.9",  # 17-800
        "zoo",
        "wdbc",
        "ionosphere",
        "lung-cancer",
        "sonar",
        "movement_libras",
        "Hill_Valley",
        "Musk version1",
        # "arrhythmia",
        "madelon",
        # "isolet5"
        ]
    
def main(max_trail=20):
    accuarcy_result = []
    columns = ['fitness', "no_fs_fitness", 'features_num', 'n_samples', "time"]
    for m in [1,2]:
        for n in [1,2,3,4]:
            init_measure = m
            gbest_measure = n
            for i in range(10, 28):
                data, labels, n_cluster = get_uci_data(i,dataset_list)
                print(data.shape)
                result_list = []
                for t in range(max_trail):
                    a_time = datetime.now()
                    fs = FSPSO(data=data, labels=labels)
                    fs.init_measure=init_measure
                    fs.gbest_measure=gbest_measure
                    fs.run()
                    b_time = datetime.now()
                    duration = (b_time - a_time).seconds
                    result_list.append([fs.gbest_fitness, fs.NO_FS_fitness, fs.FS_dim, fs.dim, duration])
                    print("trail:{:d},ARI:{:.4f} No_FS ARI:{:.4f},FS_dim:{:d},dim: {:d} time:{:d}s".
                          format(t, fs.gbest_fitness, fs.NO_FS_fitness, fs.FS_dim, fs.dim, duration))
                result_list = np.array(result_list)
                print(np.mean(result_list, axis=0))
                print(np.mean(result_list.T[1]), np.mean(result_list.T[0]), '(', np.max(result_list.T[0]), ')',
                      np.std(result_list.T[0]))
                df = pd.DataFrame(np.around(result_list, decimals=4), columns=columns)
                df.to_csv("result/" + dataset_list[i] + "-("+str(init_measure)+'-'+str(gbest_measure)+").txt", header=None, sep='\t')
                print(df)
                # print(df.describe())

def total_result():
    file_dir = "C:/Users/Administrator/Desktop/Note/wmf/7-15/FSPSO-init/"
    # file_dir = "result/"
    total_result = []
    index1=[]
    columns=['dim','Ave.ARI','Best.ARI','Std.ARI']
    for k in range(18,27):
        index2=[]
        index2.append("origin")
        index1.append(dataset_list[k])
        for i in [1,2,3,4]:
            for j in [2]:
                index2.append(str(i)+'—'+str(j))
                file=file_dir+dataset_list[k]+"-("+str(i)+'-'+str(j)+").txt"
                data = pd.read_csv(file, header=None, sep="\t").values.T
                result=[np.mean(data[3]),np.mean(data[1]),np.max(data[1]),np.std(data[1])]
                total_result.append(result)
        total_result.insert(len(total_result)-4,[np.mean(data[4]),np.mean(data[2]),np.max(data[2]),np.std(data[2])])
    print(index1,index2,len(total_result))
    index = pd.MultiIndex.from_product([index1, index2], names=['数据集', '优化算法'])
    res=pd.DataFrame(np.around(np.array(total_result), decimals=4),index=index,columns=columns)
    res.to_csv(file_dir + "FSPSO-total_result.csv", sep=",")
    print(res)
            # print(data.describe().round(4))
if __name__ == '__main__':
    # total_result()
    main()

    




















