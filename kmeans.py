import random
import numpy as np
import math
from index import assign_cluster,  update_centroids, cal_fitness,CH_index, Euclidean

class Kmeans:
    def __init__(
            self,
            n_cluster: int,
            init_pp: bool ,
            max_iter: int ,
            tolerance: float = 1e-8):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_pp = init_pp
        self.centroids = None
        self.clusters=None
        self.fitness=None
        self.CH=None

    def fit(self, data: np.ndarray, centroids:np.ndarray):
        result=[]
        self.centroids = centroids
        for i in range(self.max_iter):
            self.clusters=assign_cluster(self.centroids,data)
            new_centroid = update_centroids(self.n_cluster,self.clusters,data)
            # print(new_centroid.shape,self.centroid.shape,np.unique(self.cluster))
            diff = np.abs(self.centroids - new_centroid).sum()
            self.centroids = new_centroid
            self.fitness=cal_fitness(self.centroids,self.clusters,data)
            result.append(self.fitness)
            # print(i, self.fitness)
            if diff <= self.tolerance:
                break
        while (len(result) <self.max_iter):
            result.append(self.fitness)
        return self.fitness, self.clusters, self.centroids, result

    def _init_centroid(self, data: np.ndarray):
        if self.init_pp:
            max_score = -1e20
            best_idx = np.random.choice(range(len(data)), size=(self.n_cluster))
            for _ in range(10):
                clust = [i for i in range(0, len(data))]
                idx = [int(np.random.uniform() * len(data))]
                for _ in range(1, self.n_cluster):
                    # dist = [min([calc_vec(data[c],x) for c in idx])for x in data]
                    #采用欧几里得距离最大表示两样本相似度最大，采用样本夹角余弦，最小值会接近于0，由于精度原因效果较差
                    dist = [max([np.inner(data[c]-data[x], data[c]-data[x]) for c in idx]) for x in clust]
                    #轮盘赌方法
                    dist = np.array(dist)
                    # print(idx,dist.shape)
                    dist[np.isnan(dist)]=0
                    # print(dist)
                    dist = dist / dist.sum()
                    cumdist = np.cumsum(dist)
                    # print(cumdist)
                    prob = np.random.rand()
                    for i, c in enumerate(cumdist):
                        #print(prob,c)
                        if prob < c and i not in idx:
                            idx.append(i)
                            clust.remove(i)
                            break
                centroids = data[idx]
                cluster = assign_cluster(centroids, data)
                centroid = update_centroids(self.n_cluster,cluster,data)
                score = cal_fitness(centroid,cluster,data)
                if max_score<score:
                    best_idx=idx
                    max_score=score
            centroids = np.array([data[c] for c in best_idx])
        else:
            max_score=-1e20
            best_idx=np.random.choice(range(len(data)), size=(self.n_cluster))
            for _ in range(1):
                idx = np.random.choice(range(len(data)), size=(self.n_cluster))
                centroids = data[idx]
                cluster = assign_cluster(centroids, data)
                centroid = update_centroids(self.n_cluster,cluster,data)
                score = cal_fitness(centroid,cluster,data)
                #print("k-means initial\t",score)
                if max_score<score:
                    best_idx=idx
                    max_score=score
            centroids = data[best_idx]
            # centroids = data[[1,100,200,300]]
            # print(centroids)
        return centroids

if __name__ == "__main__":
    pass
